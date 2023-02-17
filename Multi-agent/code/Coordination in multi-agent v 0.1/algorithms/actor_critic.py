from matplotlib.pyplot import axis
import torch
import torch.nn as nn
import numpy as np
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from utils.util import get_shape_from_obs_space


def Wasserstein(mu, sigma, idx1, idx2):
    p1 = torch.sum(torch.pow((mu[idx1] - mu[idx2]), 2), 1)
    p2 = torch.sum(
        torch.pow(torch.pow(sigma[idx1], 1/2) - torch.pow(sigma[idx2], 1/2), 2), 1)
    return p1+p2


def Intention_correlation(intention, tpdv, num_agents, intention_size, causal_inference_or_kl, agent_id, intention_feature, _use_naive_recurrent_policy, _use_recurrent_policy, 
                          rnn, masks, act, available_actions, deterministic, original_action_logits, args, intention_mean, intention_std, actor_features_obs):
    #intentions = intention.rsample()
        intentions = check(intention.rsample()).to(**tpdv).view(-1, num_agents, intention_size) # 32*3, 20 -> 32, 3, 20
        # 加入因果推断或KL散度
        correlated_agents = []
        if causal_inference_or_kl: # CI
            
            for agent_idx in intentions.shape[1]:
                if agent_idx == agent_id:
                    continue
                another_intention = intentions[:, agent_idx] # 32, 20
                temp_intention = torch.zeros_like(another_intention).repeat(num_agents, 1).view(another_intention.shape[0], num_agents, intention_size) # 32, 60 -> 32, 3, 20
                temp_intention[:, agent_idx, :] = another_intention 
                temp_intention = temp_intention.view(temp_intention.shape[0], -1) # 32, 60
                actor_features = intention_feature(torch.cat([actor_features_obs, temp_intention], dim=-1))
                if _use_naive_recurrent_policy or _use_recurrent_policy:
                    actor_features, rnn_states = rnn(actor_features, rnn_states, masks)

                actions, action_log_probs, action_logits = act(actor_features, available_actions, deterministic)
                kl_prob = act.kl_divergence(original_action_logits.mean, action_logits.mean, original_action_logits.stddev, action_logits.stddev,\
                    original_action_logits, action_logits)
                #kl_prob = original_action_log_probs.mean() * torch.log(original_action_log_probs.mean() / action_log_probs.mean())
                if kl_prob > args.causal_inference_coef:
                    correlated_agents.append(agent_idx)
        else:
            for agent_idx in intentions.shape[1]:
                if agent_idx == agent_id:
                    continue
                # another_intention = check(intention).to(**tpdv).view(-1, num_agents, 1)[:, agent_idx, :].cpu().numpy() # Normal((32*3, 20), (32*3, 20)) -> 
                # another_intention = another_intention.reshape(-1)
                another_intention_mu = intention.mean.reshape(-1, num_agents, intention_size)[:, agent_idx, :] # 32, 20
                another_intention_std = intention.stddev.reshape(-1, num_agents, intention_size)[:, agent_idx, :] # 32, 20
                # another_intention = list(map(lambda x: [x.mean, x.stddev], another_intention))
                # mu, std = torch.stack(list(zip(*another_intention))[0], dim=0), torch.stack(list(zip(*another_intention))[1], dim=0)
                #another_intention = torch.from_numpy(np.stack(np.array(list(map(lambda x: [x.mean, x.stddev], another_intention)))))              
                #mu = another_intention.mean(dim=0)
                #std = another_intention.std(dim=0)
                w_distance = Wasserstein(mu=[intention_mean, another_intention_mu.mean(axis=0)], sigma=[intention_std, another_intention_std.mean(axis=0)], idx1=0, idx2=1)
                if w_distance < args.causal_inference_coef:
                    correlated_agents.append(agent_idx)
        final_intention = np.zeros((intentions.shape[0], num_agents, intention_size))
        for agent_idx in correlated_agents:
            final_intention[:, agent_idx, :] = intentions[:, agent_idx, :]
        final_intention = final_intention.view(final_intention.shape[0], -1) # 32, 60
        actor_features = intention_feature(torch.cat([actor_features_obs, final_intention], dim=-1))
        return actor_features

class Actor(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, action_space, num_agents, agent_id, device=torch.device("cpu")):
        super(Actor, self).__init__()
        self.hidden_size = args.hidden_size
        self.args=args
        self.num_agents = num_agents
        self.agent_id = agent_id
        self.intention_size = self.args.intention_size
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks  # by default True, whether to mask useless data in policy loss
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy # by default False, use the whole trajectory to calculate hidden states.
        self._use_recurrent_policy = args.use_recurrent_policy # by default, use Recurrent Policy. If set, do not use.
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.intention_generator = init_(nn.Linear(self.hidden_size, self.intention_size))
        # self.self_intention_feature = nn.Sequential(init_(nn.Linear(obs_shape[0] + self.intention_size, obs_shape[0])),
        #               nn.ReLU(),
        #               init_(nn.Linear(obs_shape[0], obs_shape[0])))
        self.intention_feature = nn.Sequential(init_(nn.Linear(self.hidden_size + (self.intention_size * num_agents), self.hidden_size)),
                      nn.ReLU(),
                      init_(nn.Linear(self.hidden_size, self.hidden_size)))

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, intention=None, use_intention=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        actor_features_obs = self.base(obs) # 32 64

        self.intention = self.intention_generator(actor_features_obs) # 32 20
        self.intention_mean = self.intention.mean(dim=0) # 20
        self.intention_std = self.intention.std(dim=0) # 20
        temp_intention = torch.zeros_like(self.intention).repeat(self.num_agents, 1).view(self.intention.shape[0], self.num_agents, self.intention_size) # 32, 20 -> 32, 60 -> 32, 3, 20
        temp_intention[:, self.agent_id, :] = self.intention
        #final_intention = temp_intention.contiguous()
        self.intention = temp_intention.contiguous().view(temp_intention.shape[0], -1) # 32, 60
        #self.intention = temp_intention.view(temp_intention.shape[0], -1) # 32, 60
        actor_features_obs = self.intention_feature(torch.cat([actor_features_obs, self.intention], dim=-1)) # 32, 64+60 -> 32, 64
        #actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features_obs, rnn_states, masks)

        actions, original_action_log_probs, original_action_logits = self.act(actor_features, available_actions, deterministic)

        if use_intention and intention is not None:
            actor_features = Intention_correlation(intention, self.tpdv, self.num_agents, self.intention_size, self.args.causal_inference_or_kl, self.agent_id, self.intention_feature, self._use_naive_recurrent_policy, self._use_recurrent_policy,
                                                   self.rnn, masks, self.act, available_actions, deterministic, original_action_logits, self.args, self.intention_mean, self.intention_std, actor_features_obs)
            # intentions = check(intention.rsample()).to(**self.tpdv).view(-1, self.num_agents, self.intention_size)
            # # TODO 加入因果推断或KL散度
            # correlated_agents = []
            # if self.args.causal_inference_or_kl: # CI
                
            #     for agent_id in intentions.shape[1]:
            #         if agent_id == self.agent_id:
            #             continue
            #         another_intention = intentions[:, agent_id]
            #         temp_intention = torch.zeros_like(another_intention).repeat(1, self.num_agents, 1)
            #         temp_intention[:, agent_id, :] = another_intention
            #         temp_intention = temp_intention.view(temp_intention.shape[0], -1)
            #         actor_features = self.intention_feature(torch.cat([actor_features, temp_intention], dim=-1))
            #         if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            #             actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

            #         actions, action_log_probs, action_logits = self.act(actor_features, available_actions, deterministic)
            #         kl_prob = self.act.kl_divergence(original_action_logits.mean, action_logits.mean, original_action_logits.stddev, action_logits.stddev,\
            #             original_action_logits, action_logits)
            #         if kl_prob > self.args.causal_inference_coef:
            #             correlated_agents.append(agent_id)
            # else:
            #     for agent_id in intentions.shape[1]:
            #         if agent_id == self.agent_id:
            #             continue
            #         another_intention = check(intention).to(**self.tpdv).view(-1, self.num_agents, 1)[:, agent_id, :].cpu().numpy()
            #         another_intention = another_intention.reshape(-1)
            #         another_intention = list(map(lambda x: [x.mean, x.stddev], another_intention))
            #         mu, std = torch.stack(list(zip(*another_intention))[0], dim=0), torch.stack(list(zip(*another_intention))[1], dim=0)
            #         w_distance = Wasserstein(mu=[self.intention_mean, mu.mean(dim=0)], sigma=[self.intention_std, std.mean(dim=0)], idx1=0, idx2=1)
            #         if w_distance < self.args.causal_inference_coef:
            #             correlated_agents.append(agent_id)
            # for agent_id in correlated_agents:
            #     final_intention[:, agent_id, :] = intentions[:, agent_id, :]
            # final_intention = final_intention.view(final_intention.shape[0], -1)
            # actor_features = self.intention_feature(torch.cat([actor_features_obs, final_intention], dim=-1))

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)
        # TODO 为了稳定智能体的训练，1. 得出的意图与之前的意图比值需要在一定范围内 2. long/short term intention实 剔除扰乱意图的动作
        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, intention=None, use_intention=False):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        # TODO intention
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features_obs = self.base(obs)

        self.intention = self.intention_generator(actor_features_obs) # 32 20
        self.intention_mean = self.intention.mean(dim=0) # 20
        self.intention_std = self.intention.std(dim=0) # 20
        temp_intention = torch.zeros_like(self.intention).repeat(self.num_agents, 1).view(self.intention.shape[0], self.num_agents, self.intention_size) # 32, 20 -> 32, 60 -> 32, 3, 20
        temp_intention[:, self.agent_id, :] = self.intention
        #final_intention = temp_intention.contiguous()
        self.intention = temp_intention.contiguous().view(temp_intention.shape[0], -1) # 32, 60
        #self.intention = temp_intention.view(temp_intention.shape[0], -1) # 32, 60
        actor_features_obs = self.intention_feature(torch.cat([actor_features_obs, self.intention], dim=-1)) # 32, 64+60 -> 32, 64
        #actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features_obs, rnn_states, masks)

        actions, original_action_log_probs, original_action_logits = self.act(actor_features, available_actions, deterministic=False)

        if use_intention and intention is not None:
            actor_features = Intention_correlation(intention, self.tpdv, self.num_agents, self.intention_size, self.args.causal_inference_or_kl, self.agent_id, self.intention_feature, self._use_naive_recurrent_policy, self._use_recurrent_policy,
                                                   self.rnn, masks, self.act, available_actions, False, original_action_logits, self.args, self.intention_mean, self.intention_std, actor_features_obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        if self.args.algorithm_name=="hatrpo":
            action_log_probs, dist_entropy, action_mu, action_std, all_probs= self.act.evaluate_actions_trpo(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs
        else:
            action_log_probs, dist_entropy, action_mu, action_std, all_probs = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy, action_mu, action_std, all_probs


class Critic(nn.Module):
    """
    Critic network class for HAPPO. Outputs value function predictions given centralized input (HAPPO) or local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, num_agents, agent_id, device=torch.device("cpu"), intention_generator=None, intention_feature=None, act=None):
        super(Critic, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        self.intention_generator = intention_generator
        self.intention_feature = intention_feature
        self.intention_size = self.args.intention_size
        self.num_agents = num_agents
        self.agent_id = agent_id
        self.act = act

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, intention=None, use_intention=False):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features_obs = self.base(cent_obs)

        self.intention = self.intention_generator(critic_features_obs)  # 32 20
        self.intention_mean = self.intention.mean(dim=0) # 20
        self.intention_std = self.intention.std(dim=0) # 20
        temp_intention = torch.zeros_like(self.intention).repeat(self.num_agents, 1).view(self.intention.shape[0], self.num_agents, self.intention_size) # 32, 20 -> 32, 60 -> 32, 3, 20
        temp_intention[:, self.agent_id, :] = self.intention
        #final_intention = temp_intention.contiguous()
        self.intention = temp_intention.contiguous().view(temp_intention.shape[0], -1) # 32, 60
        #self.intention = temp_intention.view(temp_intention.shape[0], -1) # 32, 60
        critic_features_obs = self.intention_feature(torch.cat([critic_features_obs, self.intention], dim=-1)) # 32, 64+60 -> 32, 64
        #actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features_obs, rnn_states, masks)

        actions, original_action_log_probs, original_action_logits = self.act(critic_features, None, deterministic=False)

        if use_intention and intention is not None:
            critic_features = Intention_correlation(intention, self.tpdv, self.num_agents, self.intention_size, self.args.causal_inference_or_kl, self.agent_id, self.intention_feature, self._use_naive_recurrent_policy, self._use_recurrent_policy,
                                                   self.rnn, masks, self.act, None, False, original_action_logits, self.args, self.intention_mean, self.intention_std, critic_features_obs)


        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states
