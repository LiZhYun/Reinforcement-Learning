import argparse
import imp
import os
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.ppo.model_train import Policy
from gym.spaces import Box
from agents.ppo.ppo import PPO
from agents.ppo.storage import RolloutStorage

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
# TODO 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

actor_critic = Policy(
        (1, 40, 40),
        [Box(-100, 200, shape=(1,)), Box(-30, 30, shape=(1,))],
        base_kwargs={'recurrent': True})
actor_critic.to(device)
net = os.path.dirname(os.path.abspath(__file__)) + '/olympics-integrated.pt'
actor_critic = torch.load(net).to(device)
agent = PPO(
            actor_critic,
            0.2,
            3,
            1,
            0.5,
            0.01,
            lr=5e-5,
            eps=1e-5,
            max_grad_norm=0.1)

rollouts = RolloutStorage(1500,
                        (1, 40, 40), 2,
                        actor_critic.recurrent_hidden_state_size)
rollouts.to(device)
eval_recurrent_hidden_states = torch.zeros(
        actor_critic.recurrent_hidden_state_size, device=device)
eval_masks = torch.zeros(1, device=device)

def my_controller(observation, action_space, is_act_continuous=False):
    value, [action1, action2], [action_log_prob1, action_log_prob2], eval_recurrent_hidden_states = agent.actor_critic.act(
                observation['obs']['agent_obs'].reshape(-1), eval_recurrent_hidden_states,
                eval_masks)

    return value, [action1, action2], [action_log_prob1, action_log_prob2], eval_recurrent_hidden_states
