import imp
import torch
from torch.distributions import kl_divergence
import numpy as np
from collections import defaultdict
from utils.util import check, get_shape_from_obs_space, get_shape_from_act_space
from algorithms.encoder_decoder import build_input
import copy

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:]) # episode_length, n_rollout_threads

def _cast(x):
    return x.transpose(1,0,2).reshape(-1, *x.shape[2:])

class SeparatedReplayBuffer(object):
    # obs_space: [all_feats, [n_allies, n_ally_feats], [n_enemies, n_enemy_feats], [1, move_feats], [1, own_feats+agent_id_feats+timestep_feats]]
    # act_space: Discrete(self.n_actions)
    def __init__(self, args, obs_space, share_obs_space, act_space, n_agents):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.rnn_hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N  # 下一层的hidden state作为上一层的输入
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda   # weighted combination of n-step returns
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart # 归一化
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.clip_param = args.clip_param
        self.n_agents = n_agents
        self.args = args

        self.tpdv = dict(dtype=torch.float32, device="cuda")

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, act_space.n), dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.last_actions = [[[] for k in range(self.n_agents)] for j in range(self.n_rollout_threads)]
        # self.last_actions = np.zeros((self.episode_length, self.n_rollout_threads, self.n_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        
        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        

        self.factor = None

        self.step = 0

    def update_factor(self, factor):
        self.factor = factor.copy()

    def insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None, all_last_actions=None):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()
        if all_last_actions is not None:
            for env_idx in range(self.n_rollout_threads):
                for agent_idx in range(self.n_agents):
                    if all_last_actions[env_idx, agent_idx] != 0.:
                        self.last_actions[env_idx][agent_idx].append(all_last_actions[env_idx, agent_idx])
                        #self.last_actions[self.step + 1, env_idx, agent_idx] = all_last_actions[env_idx, agent_idx].copy()
        # if self.step + 1 == self.episode_length:
        #     self.last_actions = [[[] for k in range(self.n_agents)] for j in range(self.n_rollout_threads)] # 更新后重置
        self.step = (self.step + 1) % self.episode_length


    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None, all_last_actions=None):
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()
        if all_last_actions is not None:

            self.last_actions[self.step + 1] = all_last_actions.copy()

        self.step = (self.step + 1) % self.episode_length
    
    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()
        if self.last_actions is not None:
            # self.last_actions[0] = self.last_actions[-1].copy()
            self.last_actions = [[[] for k in range(self.n_agents)] for j in range(self.n_rollout_threads)] # 更新后重置
        

    def chooseafter_update(self):
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        use proper time limits, the difference of use or not is whether use bad_mask
        """
        if self._use_proper_time_limits:  # compute returns taking into account time limits 是否是最后一个时刻
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0    # Generalized advantage estimation
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[
                            step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (  # minibatch size M <= NT
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        if self.factor is not None:
            # factor = self.factor.reshape(-1,1)
            factor = self.factor.reshape(-1, self.factor.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            if self.factor is None:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
            else:
                factor_batch = factor[indices]
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        n_rollout_threads = self.rewards.shape[1]
        assert n_rollout_threads >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_mini_batch))
        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(self.share_obs[:-1, ind])
                obs_batch.append(self.obs[:-1, ind])
                rnn_states_batch.append(self.rnn_states[0:1, ind])
                rnn_states_critic_batch.append(self.rnn_states_critic[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(self.available_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                active_masks_batch.append(self.active_masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.factor is not None:
                    factor_batch.append(self.factor[:,ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            if self.factor is not None:
                factor_batch=np.stack(factor_batch,1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, -1) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch, 1).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch, 1).reshape(N, *self.rnn_states_critic.shape[2:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch=_flatten(T,N,factor_batch)
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)
            if self.factor is not None:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch
            else:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length, intention_buffer, encoder_decoder, long_short_clip=False):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length # 32 * 200
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L] # 6400 / 10 = 640
        mini_batch_size = data_chunks // num_mini_batch # 640 / 1 = 640

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, episode_length, data_chunk_length))
        assert data_chunks >= 2, ("need larger batch size")

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 3:
            share_obs = self.share_obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.share_obs.shape[2:])
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions) # 6400, 1
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        for env_idx in range(self.n_rollout_threads):
            for agent_idx in range(self.n_agents):
                if len(self.last_actions[env_idx][agent_idx]) < self.episode_length:
                    self.last_actions[env_idx][agent_idx].extend([0] * (self.episode_length - len(self.last_actions[env_idx][agent_idx])))
        
        # last_actions = np.array(self.last_actions)[:, :, :, -1] # 32, 3, 200, 1
        # last_actions = last_actions.transpose(0, 2, 1, 3) # 32, 3, 200, 1 -> 32, 200, 3, 1
        # last_actions = self.last_actions.transpose(1, 0, 2, 3) # 200, 32, 3, 1 -> 32, 200, 3, 1
        if self.factor is not None:
            factor = _cast(self.factor)
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states_critic.shape[2:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []
            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
                obs_batch.append(obs[ind:ind+data_chunk_length])
                actions_batch.append(actions[ind:ind+data_chunk_length]) # 640, 10, 1
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind+data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind+data_chunk_length])
                return_batch.append(returns[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind+data_chunk_length])
                adv_targ.append(advantages[ind:ind+data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                # 不是ind:ind+data_chunk_length
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
                if self.factor is not None:
                    factor_batch.append(factor[ind:ind+data_chunk_length])
            L, N = data_chunk_length, mini_batch_size # 10, 640

            # These are all from_numpys of size (N, L, Dim)
            share_obs_batch = np.stack(share_obs_batch) # 640 10 64 -> np_array 640, 10, 64
            obs_batch = np.stack(obs_batch)

            actions_batch = np.stack(actions_batch)  # 640, 10, 1
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch)
            if self.factor is not None:
                factor_batch = np.stack(factor_batch)
            value_preds_batch = np.stack(value_preds_batch)
            return_batch = np.stack(return_batch)
            masks_batch = np.stack(masks_batch)
            active_masks_batch = np.stack(active_masks_batch)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch)
            adv_targ = np.stack(adv_targ)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:]) # 640, 1, 64 -> 640, 1, 64
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[2:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)  # (6400, 64)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)  # (6400, 1)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(L, N, factor_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ) # (640, 10, 1) -> 6400, 1
            # TODO 生成action_traj
            
            if long_short_clip:
                last_action_clipped = [[[] for k in range(self.n_agents)] for j in range(self.n_rollout_threads)]
                for env_idx in range(self.n_rollout_threads):
                    for agent_idx in range(self.n_agents):
                        if len(self.last_actions[env_idx][agent_idx]) > 30:
                            for idx in range(0, len(self.last_actions[env_idx][agent_idx]), 3):
                                if 0. in self.last_actions[env_idx][agent_idx][idx:idx+3]:
                                    last_action_clipped[env_idx][agent_idx].extend(self.last_actions[env_idx][agent_idx][idx:])
                                    break
                                else:
                                    # 1, 3, 1
                                    temp_actions = check(np.array(self.last_actions[env_idx][agent_idx][idx:idx+3])).to(**self.tpdv).unsqueeze(0).unsqueeze(-1)
                                    intention, _ = encoder_decoder.encoder(temp_actions) # Normal((1, 20), (1, 20))
                                    kl_ratio = kl_divergence(intention, intention_buffer)
                                    # kl_ratio = torch.mean(check(intention.rsample()).to(**self.tpdv).mean(dim=0) / check(intention_buffer.rsample()).to(**self.tpdv).mean(dim=0))
                                    if kl_ratio < self.args.causal_inference_coef:
                                        last_action_clipped[env_idx][agent_idx].extend(self.last_actions[env_idx][agent_idx][idx:idx+3])
                        else:
                            last_action_clipped[env_idx][agent_idx].extend(self.last_actions[env_idx][agent_idx])
                    if len(last_action_clipped[env_idx][agent_idx]) < self.episode_length:
                        last_action_clipped[env_idx][agent_idx].extend([0] * (self.episode_length - len(last_action_clipped[env_idx][agent_idx])))
                

                last_actions = np.array(last_action_clipped)[:, :, :, -1]  # 32, 3, 200, 1
                last_actions = last_actions.transpose(0, 2, 1, 3) # # 32, 200, 3, 1
                # intention_mask = np.ones((self.n_rollout_threads, self.episode_length, self.n_agents, 1))
                # for idx in range(0, episode_length, 3):
                #     temp_actions = check(build_input(last_actions[:, idx:idx+3])).to(**self.tpdv)
                #     intention, _ = encoder_decoder.encoder(temp_actions)
                #     ratio = torch.mean(check(intention.rsample()).to(**self.tpdv).mean(dim=0) / check(intention_buffer.rsample()).to(**self.tpdv).mean(dim=0))
                #     if ratio > 1.0 + self.clip_param or ratio < 1.0 - self.clip_param:
                #         intention_mask[:, idx:idx+3] = np.zeros((3, self.n_agents, 1), dtype=np.float32)
                #clipped_last_actions = last_actions * intention_mask
                last_actions_past = last_actions[:, :160]
                last_actions_future = last_actions[:, 160:]
            else:
                last_actions = np.array(last_actions)[:, :, :, -1]  # 32, 3, 200, 1
                last_actions = last_actions.transpose(0, 2, 1, 3) # # 32, 200, 3, 1
                last_actions_past = last_actions[:, :160]
                last_actions_future = last_actions[:, 160:]
            # pad_data = np.zeros(shape=(self.max_seq_len, data.shape[1]))
            # pad_data[0:data.shape[0]] = data
            if self.factor is not None:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch, last_actions_past, last_actions_future
            else:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, last_actions_past, last_actions_future
