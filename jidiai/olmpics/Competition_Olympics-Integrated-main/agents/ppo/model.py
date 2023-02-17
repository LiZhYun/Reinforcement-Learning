import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from distributions import Bernoulli, Categorical, DiagGaussian, MultiDiagGaussian
from ppo_utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        self.action_space = action_space
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            num_first_output = action_space[0].shape[0]
            num_sec_output = action_space[1].shape[0]
            self.dist = MultiDiagGaussian(self.base.output_size, [num_first_output, num_sec_output])
            

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)
        # TODO .to(torch.device('cuda'))
        if deterministic:
            action1 = np.clip(dist[0].mode().cpu(), self.action_space[0].low, self.action_space[0].high).to(torch.device('cpu'))
            action2 = np.clip(dist[1].mode().cpu(), self.action_space[1].low, self.action_space[1].high).to(torch.device('cpu'))
        else:
            action1 = np.clip(dist[0].sample().cpu(), self.action_space[0].low, self.action_space[0].high).to(torch.device('cpu'))
            action2 = np.clip(dist[1].sample().cpu(), self.action_space[1].low, self.action_space[1].high).to(torch.device('cpu'))

        action_log_probs1 = dist[0].log_probs(action1)
        dist_entropy1 = dist[0].entropy().mean()
        action_log_probs2 = dist[1].log_probs(action2)
        dist_entropy2 = dist[1].entropy().mean()

        return value, [action1, action2], [action_log_probs1, action_log_probs2], rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):

        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        assert torch.any(torch.isnan(actor_features)) == False
        dist = self.dist(actor_features)

        action_log_probs1 = dist[0].log_probs(action[:, 0])
        dist_entropy1 = dist[0].entropy().mean()
        action_log_probs2 = dist[1].log_probs(action[:, 1])
        dist_entropy2 = dist[1].entropy().mean()

        return value, [action_log_probs1, action_log_probs2], [dist_entropy1, dist_entropy2], rnn_hxs
    
    def load(self):
        net = os.path.dirname(os.path.abspath(__file__)) + '/olympics-integrated.pt'
        return torch.load(net)


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 1e-3)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        assert torch.any(torch.isnan(x)) == False, "input is nan"
        if len(x.shape) == 1:
            x, hxs = self.gru(x.unsqueeze(0).unsqueeze(0), (hxs * masks).unsqueeze(0).unsqueeze(0))
            x = x.squeeze(0).squeeze(0)
            hxs = hxs.squeeze(0).squeeze(0)
        else:
            # # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # # unflatten
            x = x.view(T, N, x.size(1))

            # # Same deal with masks
            masks = masks.view(T, N)

            # # Let's figure out which steps in the sequence have a zero for any agent
            # # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                assert torch.any(torch.isnan(x)) == False, "input is nan"
                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))
                assert torch.any(torch.isnan(rnn_scores)) == False, "input is nan"
                outputs.append(rnn_scores)

            # # assert len(outputs) == T
            # # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)
        assert torch.any(torch.isnan(hxs)) == False, "input is nan"
        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=128):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.xavier_uniform_, lambda x: nn.init.
                               constant_(x, 5), nn.init.calculate_gain('tanh'))

        # self.main1 = init_(nn.Conv2d(num_inputs, 32, 4, stride=4))
        # self.act1 = nn.Tanh()
        # self.main2 = init_(nn.Conv2d(32, 64, 4, stride=2))
        # self.act2 = nn.Tanh()
        # self.main3 = init_(nn.Conv2d(64, 32, 3, stride=1))
        # self.act3 = nn.Tanh()
        # self.flat1 = Flatten()
        # self.main4 = init_(nn.Linear(32 * 2 * 2, hidden_size))
        # self.act4 = nn.Tanh()
        self.main = nn.Sequential(
            #init_(nn.Conv2d(num_inputs, 32, 4, stride=4)), nn.Tanh(),
            #init_(nn.Conv2d(num_inputs, 8, 4, stride=4)), nn.Tanh(),
            init_(nn.Conv2d(num_inputs, 4, 8, stride=8)), nn.Tanh(), Flatten(),
            init_(nn.Linear(4 * 5 * 5, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        #self.feature_norm = nn.LayerNorm((40, 40), eps=1e-2)
        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        inputs = torch.from_numpy(inputs) if type(inputs) == np.ndarray else inputs
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0)
        #y = self.feature_norm(inputs)
        #x = self.main(y)
        inputs = inputs / 255.0
        x = self.main(inputs.to(torch.float))
        # x1 = self.main1(inputs)
        # x2 = self.act1(x1)
        # x3 = self.main2(x2)
        # x4 = self.act2(x3)
        # x5 = self.main3(x4)
        # x6 = self.act3(x5)
        # x7 = self.flat1(x6)
        # x8 = self.main4(x7)
        # x9 = self.act4(x8)
        assert torch.any(torch.isnan(x)) == False, "input is nan"
        x = x.squeeze(0)

        if self.is_recurrent:
            
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        assert torch.any(torch.isnan(rnn_hxs)) == False, "input is nan"
        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
