import torch
import torch.nn as nn
from algorithms.utils.util import init, check
from algorithms.utils.cnn import CNNBase
from algorithms.utils.mlp import MLPBase
from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer
from utils.util import get_shape_from_act_space
import numpy as np
from torch.distributions import Normal

def build_input(past, future=None):
    if future:
        actions = np.concatenate([past, future], axis=1)
    else:
        actions = past
    actions = actions.transpose(0, 2, 1, 3)  # [bz, n_agents, timesteps, feat]
    actions = actions.reshape((-1, *actions.shape[2:])) # [batch_size * n_agents, timesteps, input_size]

    return actions

class Encoder_Decoder(nn.Module):
    def __init__(self, args, act_space, device=torch.device("cpu")):
        super(Encoder_Decoder, self).__init__()
        self.encoder = Encoder(args, act_space, device)
        self.decoder = Decoder(args, act_space, device)

class Encoder(nn.Module):
    """
    Actor network class for HAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, action_space, device=torch.device("cpu")):
        super(Encoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.intention_size = args.intention_size
        self.args=args
        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks  # by default True, whether to mask useless data in policy loss
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy # by default False, use the whole trajectory to calculate hidden states.
        self._use_recurrent_policy = args.use_recurrent_policy # by default, use Recurrent Policy. If set, do not use.
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        rnn_input_dim = get_shape_from_act_space(action_space)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = nn.GRU(rnn_input_dim, self.hidden_size, num_layers=self._recurrent_N)
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    if self._use_orthogonal:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.xavier_uniform_(param)
            #self.rnn = RNNLayer(rnn_input_dim, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        
        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(self.hidden_size)

        self.zI_mu_enc = init_(nn.Linear(self.hidden_size, self.intention_size))
        self.zI_std_enc = nn.Sequential(
            init_(nn.Linear(self.hidden_size, self.intention_size)), nn.Softplus())

        self.to(device)

    def forward(self, actions, hidden=None):
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
        # actions shape: [batchsize * n_agents, num_timesteps, num_dims] -> [num_timesteps, batchsize * n_agents, num_dims]
        actions = check(actions).to(**self.tpdv)
        actions = actions.transpose(1, 0)

        if hidden is None:
            _, temp_rnn_state = self.rnn(actions)  # (1, batchsize * n_agents, hidden_size)
        else:
            hidden = check(hidden).to(**self.tpdv)
            _, temp_rnn_state = self.rnn(actions, hidden)
        x = temp_rnn_state.squeeze(0)

        if self._use_feature_normalization:
            x = self.feature_norm(x)

        zI_mu = self.zI_mu_enc(x)  # (batchsize * n_agents, z_dim)
        zI_std = self.zI_std_enc(x)
        
        return Normal(zI_mu, zI_std), temp_rnn_state


class Decoder(nn.Module):
    """
    Critic network class for HAPPO. Outputs value function predictions given centralized input (HAPPO) or local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, action_space, device=torch.device("cpu")):
        super(Decoder, self).__init__()
        self.hidden_size = args.hidden_size
        self.intention_size = args.intention_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        input_dim = get_shape_from_act_space(action_space)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.x_enc = init_(nn.Linear(input_dim, self.hidden_size))
        self.out_fc = nn.Sequential(init_(nn.Linear(self.hidden_size + self.intention_size, self.hidden_size)),
                                    nn.ReLU(),
                                    init_(nn.Linear(self.hidden_size, input_dim)))

        self.to(device)

    def forward(self, intention, inputs, pred_steps=40):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        inputs = check(inputs).to(**self.tpdv)
        intention = check(intention).to(**self.tpdv)
        inputs = inputs.transpose(0, 1).contiguous() # [num_timesteps, batchsize * n_agents, num_dims]
        sizes = inputs.shape
        preds_out = torch.zeros(sizes).to(**self.tpdv)
        last_pred = inputs[0::pred_steps, :, :]
        for b_idx in range(0, last_pred.shape[0]):
            x = last_pred[b_idx]
            for step in range(0, pred_steps):
                x = self.x_enc(x)
                h = torch.cat([x, intention], dim=-1)
                x_pred = self.out_fc(h)  # [batchsize * n_agents, num_dims]
                preds_out[step + b_idx * pred_steps, :, :] = x_pred

        return preds_out.transpose(0, 1).contiguous() # [batch_size * num_vars, num_timesteps, num_inputs]
