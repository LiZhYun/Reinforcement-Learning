import argparse
import imp
import os
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Policy
from ppo import PPO
from gym.spaces import Box
#from agents.ppo.storage import RolloutStorage

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
# TODO 

device = "cpu"

actor_critic = Policy(
        (1, 40, 40),
        [Box(-100, 200, shape=(1,)), Box(-30, 30, shape=(1,))],
        base_kwargs={'recurrent': True})
actor_critic.to(device)

net = os.path.dirname(os.path.abspath(__file__)) + '/pretrain_actor_critic.pt'
pretrain_actor_critic = torch.load(net)

# torch.save(pretrain_actor_critic.state_dict(), "./pretrain_actor_critic.pt")
model_dict = actor_critic.state_dict()
pretrained_dict = {k: v for k, v in pretrain_actor_critic.items() if k in model_dict}
model_dict.update(pretrained_dict)
actor_critic.load_state_dict(model_dict)
agent = PPO(
            actor_critic,
            0.2,
            4,
            1,
            0.2,
            0.01,
            lr=5e-4,
            eps=1e-5,
            max_grad_norm=0.2)
#print("load")
#rollouts.to(device)
eval_recurrent_hidden_states = torch.zeros(
        actor_critic.recurrent_hidden_state_size, device=device)
eval_masks = torch.zeros(1, device=device)

def my_controller(observation, action_space, is_act_continuous=False):
        global eval_recurrent_hidden_states, eval_masks

        value, [action1, action2], [action_log_prob1, action_log_prob2], eval_recurrent_hidden_states = agent.actor_critic.act(
                observation['obs']['agent_obs'].reshape(-1, 40, 40), eval_recurrent_hidden_states,
                eval_masks)

        return [action1.cpu().numpy(), action2.cpu().numpy()]
