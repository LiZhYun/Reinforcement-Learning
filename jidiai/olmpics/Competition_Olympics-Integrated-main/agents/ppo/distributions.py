import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppo_utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class MultiDiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MultiDiagGaussian, self).__init__()
        self.num_outputs = num_outputs

        init_0 = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        init_1 = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean0 = init_0(nn.Linear(num_inputs, num_outputs[0]))
        self.fc_mean1 = init_1(nn.Linear(num_inputs, num_outputs[1]))
        self.fc_std0 = nn.Sequential(init_0(nn.Linear(num_inputs, num_outputs[0])), nn.Softplus())
        #init_0(nn.Linear(num_inputs, num_outputs[0]))
        self.fc_std1 = nn.Sequential(init_1(nn.Linear(num_inputs, num_outputs[1])), nn.Softplus())
        #action_logstd0 = nn.Parameter(torch.ones(num_outputs[0])).to(torch.device('cuda'))
        #action_logstd1 = nn.Parameter(torch.ones(num_outputs[1])).to(torch.device('cuda'))
        #self.logstd = AddBias(torch.zeros(sum(num_outputs)))
        #self.to(torch.device('cpu'))

    def forward(self, x):
        action_mean0 = self.fc_mean0(x)
        action_mean1 = self.fc_mean1(x)
        action_std0 = self.fc_std0(x)
        action_std1 = self.fc_std1(x)
        #action_means = torch.split(action_mean, tuple(self.num_outputs), dim=-1)

        #  An ugly hack for my KFAC implementation.
        # zeros = torch.zeros(action_mean.size())
        # if x.is_cuda:
        #     zeros = zeros.cuda()

        #action_logstd0 = nn.Parameter(torch.ones(action_mean0.size())).to(torch.device('cuda'))
        #action_logstd1 = nn.Parameter(torch.ones(action_mean1.size())).to(torch.device('cuda'))
        #action_logstds = torch.split(action_logstd, tuple(self.num_outputs), dim=-1)

        return [FixedNormal(action_mean0, action_std0.exp()), FixedNormal(action_mean1, action_std1.exp())]
        # return [FixedNormal(action_mean, action_logstd.exp()) for action_mean, action_logstd in zip(action_means, action_logstds)]


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
