import sys
from pathlib import Path
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
print(sys.path)
from olympics_engine.generator import create_scenario
import argparse
from olympics_engine.agent import *
import time

from scenario import Running, table_hockey, football, wrestling, billiard, curling, curling_joint

from AI_olympics import AI_Olympics

import random
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.ppo.model import Policy
from agents.ppo.storage import RolloutStorage

def store(record, name):

    with open('logs/'+name+'.json', 'w') as f:
        f.write(json.dumps(record))

def load_record(path):
    file = open(path, "rb")
    filejson = json.load(file)
    return filejson

RENDER = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', default="all", type= str,
                        help = 'running/table-hockey/football/wrestling/billiard/curling/all')
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    for i in range(1):
        if args.map != 'all':
            Gamemap = create_scenario(args.map)
        #game = table_hockey(Gamemap)
        if args.map == 'running':
            game = Running(Gamemap)
            agent_num = 2
        elif args.map == 'table-hockey':
            game = table_hockey(Gamemap)
            agent_num = 2
        elif args.map == 'football':
            game = football(Gamemap)
            agent_num = 2
        elif args.map == 'wrestling':
            game = wrestling(Gamemap)
            agent_num = 2
        # elif args.map == 'volleyball':
        #     game = volleyball(Gamemap)
        #     agent_num = 2
        elif args.map == 'billiard':
            game = billiard(Gamemap)
            agent_num = 2
        elif args.map == 'curling':
            game = curling(Gamemap)
            agent_num = 2

        elif args.map == 'curling-joint':
            game = curling_joint(Gamemap)
            agent_num = 2

        elif args.map == 'all':
            game = AI_Olympics(random_selection = False, minimap=True)
            agent_num = 2

        actor_critic = Policy(
            game.observation_space.shape,
            game.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
        actor_critic.to(device)

        agent = random_agent()
        rand_agent = random_agent()

        obs = game.reset()
        done = False
        step = 0
        if RENDER:
            game.render()

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        time_epi_s = time.time()
        while not done:
            step += 1

            # print('\n Step ', step)

            #action1 = [100,0]#agent.act(obs)
            #action2 = [100,0] #rand_agent.act(obs)
            if agent_num == 2:
                action1, action2 = agent.act(obs[0]), rand_agent.act(obs[1])
                # action1 = [200,20]
                action1 =[200,1]
                action = [action1, action2]
            elif agent_num == 1:
                action1 = agent.act(obs)
                action = [action1]

            # if step <= 5:
            #     action = [[200,0]]
            # else:
            #     action = [[0,0]]
            # action = [[200,action1[1]]]

            obs, reward, done, _ = game.step(action)
            # print(f'reward = {reward}')
            # print('obs = ', obs)
            # plt.imshow(obs[0])
            # plt.show()
            if RENDER:
                game.render()

            # time.sleep(0.02)


        print("episode duration: ", time.time() - time_epi_s, "step: ", step, (time.time() - time_epi_s)/step)
        if args.map == 'billiard':
            print('reward =', game.total_reward)
        else:
            print('reward = ', reward)
        # if R:
        #     store(record,'bug1')

