import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.popart import PopArt
from algorithms.utils.util import check
from algorithms.encoder_decoder import build_input
from torch.distributions import Normal, kl_divergence

class HAPPO():
    """
    Trainer class for HAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (HAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 agent_id,
                 device=torch.device("cpu")):
        self.tau = args.intention_update_beta
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents
        self.agent_id = agent_id

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        self.intention_size = args.intention_size

        
        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss
    
    def get_intention_loss(self, q_actions, p_actions, q_res, p_res):
        q_action_pred = q_res["action_pred"][:, :-1, :] # [batch_size * num_vars, num_timesteps, num_inputs]
        q_target = q_actions[:, 1:, :]
        q_zI_rv = q_res["zI_rv"]
        p_zI_rv = p_res["zI_rv"]
        predict_rv = Normal(q_action_pred, self.policy.sigma_action)
        
        loss_nll = - predict_rv.log_prob(q_target).reshape(-1, self.num_agents, *q_action_pred.shape[1:]) # [batch_size, num_vars, num_timesteps, num_inputs]
        kl_zI = kl_divergence(q_zI_rv, p_zI_rv).reshape(-1, self.num_agents, self.intention_size)  # [batch_size, num_vars, intention_size]

        b_s = torch.sum(loss_nll, (1, 2, 3))  # batch_size
        loss_nll = b_s.mean() / self.num_agents
        loss_kl = kl_zI.sum(2).mean()
        loss = loss_nll + loss_kl

        return {
            "total_loss": loss.mean(),
            "loss_nll": loss_nll.mean(),
            "loss_kl": loss_kl.mean(),
            'kl_zI': kl_zI.mean()
        }


    def train_intention(self, past, future, intention_buffer, intention_clip=False):
        q_actions = check(build_input(past, future)).to(**self.tpdv)
        p_actions = check(build_input(past)).to(**self.tpdv) # [batch_size * n_agents, timesteps, input_size]
        q_res = self.policy.get_intention(q_actions)
        p_res = self.policy.get_intention(p_actions)
        loss_dict = self.get_intention_loss(q_actions, p_actions, q_res, p_res)
        loss = loss_dict["total_loss"]

        old_param = self.policy.endecoder.state_dict().clone().detach()
        prev_endecoder_para = self.policy.endecoder.state_dict().clone().detach()

        self.policy.encoder_decoder_optimizer.zero_grad()
        loss.backward()
        self.policy.encoder_decoder_optimizer.step()

        # soft update
        for target_param, param in zip(old_param, self.policy.encoder_decoder.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        model_dict = self.policy.encoder_decoder.state_dict()
        target_dict = {k: v for k, v in old_param.items()
                        if k in model_dict}
        model_dict.update(target_dict)
        self.policy.encoder_decoder.load_state_dict(model_dict)

        curr_intention = self.policy.get_intention(q_actions)["zI_rv"]

        if intention_clip:
            ratio = torch.zeros((q_actions.shape[0], self.intention_size))
            for _ in range(100):
                intention_sample = intention_buffer.rsample()
                ratio += torch.exp(curr_intention.log_prob(intention_sample) - intention_buffer.log_prob(intention_sample))
            ratio = (ratio / 100).mean()
        #ratio = torch.mean(check(curr_intention.rsample()).to(**self.tpdv).mean(dim=0) / check(intention_buffer.rsample()).to(**self.tpdv).mean(dim=0))
        
            if ratio > 1.0 + self.clip_param or ratio < 1.0 - self.clip_param:
                self.policy.endecoder.load_state_dict(prev_endecoder_para)
                curr_intention = self.policy.get_intention(q_actions)["zI_rv"]

        return loss, curr_intention  # Normal(batchsize * n_agents, z_dim)

        
    def ppo_update(self, sample, intention_buffer, update_actor=True, intention_clip=False):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, factor_batch, last_actions_past, last_actions_future = sample  # 6400, (64, 64, (1, 64), (1, 64), 1, 1, 1, 1, 1, 1, 1, 9, 1)



        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)


        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)


        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        factor_batch = check(factor_batch).to(**self.tpdv)
        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, action_mu, action_std, all_probs = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch, intention=intention_buffer, use_intention=True)
        # intention update                                                                      
        intention_loss, curr_intention = self.train_intention(last_actions_past, last_actions_future, intention_buffer, intention_clip=intention_clip)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        intention_ratio = torch.zeros((last_actions_past.shape[0], self.intention_size))
        for _ in range(100):
            intention_sample = intention_buffer.rsample()
            ratio += torch.exp(curr_intention.log_prob(intention_sample) - intention_buffer.log_prob(intention_sample))
        intention_ratio = (ratio / 100).mean()
        #intention_ratio = torch.mean(check(curr_intention.rsample()).to(**self.tpdv).mean(dim=0) / check(intention_buffer.rsample()).to(**self.tpdv).mean(dim=0))
        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(intention_ratio * factor_batch * torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(intention_ratio * factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, intention_loss, curr_intention

    def train(self, buffer, intention_buffer, intention_clip=False, long_short_clip=False, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['intention_loss'] = 0
        train_info['curr_intention'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length, intention_buffer, self.policy.encoder_decoder, long_short_clip=long_short_clip)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, intention_loss, curr_intention = self.ppo_update(sample, intention_buffer, update_actor=update_actor, intention_clip=intention_clip)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['intention_loss'] += intention_loss.mean()
                

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        train_info['curr_intention'] = curr_intention
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
