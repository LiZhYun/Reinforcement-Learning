from ast import arg
import imp
import torch
from algorithms.actor_critic import Actor, Critic
from algorithms.encoder_decoder import Encoder_Decoder, build_input
from utils.util import update_linear_schedule
from torch import nn


class HAPPO_Policy:
    """
    HAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for HAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, agent_id, device=torch.device("cpu")):
        self.args=args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.num_agents = num_agents
        self.agent_id = agent_id

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = Actor(args, self.obs_space, self.act_space, self.num_agents, self.agent_id, self.device)

        ######################################Please Note#########################################
        #####   We create one critic for each agent, but they are trained with same data     #####
        #####   and using same update setting. Therefore they have the same parameter,       #####
        #####   you can regard them as the same critic.                                      #####
        ##########################################################################################
        self.critic = Critic(args, self.share_obs_space, self.num_agents, self.agent_id, self.device, self.actor.intention_generator, self.actor.intention_feature, self.actor.act)

        self.encoder_decoder = Encoder_Decoder(args, self.act_space, self.device)

        # self.encoder = Encoder(args, self.act_space, self.device)

        # self.decoder = Decoder(args, self.act_space, self.device)

        self.sigma_action = nn.Parameter(torch.tensor(
            [1e-1]).clamp(min=1e-3)).to(self.device)



        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.encoder_decoder_optimizer = torch.optim.Adam(self.encoder_decoder.parameters(), lr=self.lr,
                                                          eps=self.opti_eps,
                                                          weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, intention=None, use_intention=False,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        # intention为(batchsize * num_atoms, z_dim)
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                    rnn_states_actor,
                                                                    masks,
                                                                    available_actions,
                                                                    deterministic, intention, use_intention)

        values, rnn_states_critic = self.critic(
            cent_obs, rnn_states_critic, masks, intention=intention, use_intention=use_intention)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks, intention=None, use_intention=False):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        #TODO 把intention也加入到critic中
        values, _ = self.critic(cent_obs, rnn_states_critic, masks, intention=intention, use_intention=use_intention)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, intention=None, use_intention=False):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        # TODO 也需要加intention
        action_log_probs, dist_entropy, action_mu, action_std, all_probs = self.actor.evaluate_actions(obs,
                                                                rnn_states_actor,
                                                                action,
                                                                masks,
                                                                available_actions,
                                                                active_masks, intention=intention, use_intention=use_intention)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy, action_mu, action_std, all_probs


    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, intention=None, use_intention=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic, intention, use_intention)
        return actions, rnn_states_actor

    def get_intention(self, actions):
        zI_rv, temp_rnn_state = self.encoder_decoder.encoder(actions)
        zI = zI_rv.rsample()
        action_pred = self.encoder_decoder.decoder(zI, actions)

        res = {
            "action_pred": action_pred,
            "zI_rv": zI_rv,  # Normal(batchsize * n_agents, z_dim)
            "zI": zI,
            "temp_rnn_state": temp_rnn_state
        }

        return res
        # 全部训练完成之后，将最后的intention保存起来，供执行时初始化intentions
        # 执行阶段，未观察到other agents动作时，先使用初始化的intention
        # 观察到other agent动作后，用intention网络生成新的intention替换掉buffer中的intention，或者使用软更新 (1-β)*old_intention + β*new_intention
        # 如果允许的话还可以在执行期间训练。
