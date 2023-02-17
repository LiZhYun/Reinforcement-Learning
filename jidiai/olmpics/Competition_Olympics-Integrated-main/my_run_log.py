# -*- coding:utf-8  -*-
from asyncore import readwrite
import os
import time
import json
import numpy as np
from collections import deque
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
sys.path.append("./olympics_engine")

from env.chooseenv import make
from utils.get_logger import get_logger
from env.obs_interfaces.observation import obs_type
from agents.ppo import submission_train as submission
from agents.ppo import ppo_utils


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_players_and_action_space_list(g):
    if sum(g.agent_nums) != g.n_player:
        raise Exception("agent number = %d 不正确，与n_player = %d 不匹配" % (sum(g.agent_nums), g.n_player))

    n_agent_num = list(g.agent_nums)
    for i in range(1, len(n_agent_num)):
        n_agent_num[i] += n_agent_num[i - 1]

    # 根据agent number 分配 player id
    players_id = []
    actions_space = []
    for policy_i in range(len(g.obs_type)):
        if policy_i == 0:
            players_id_list = range(n_agent_num[policy_i])
        else:
            players_id_list = range(n_agent_num[policy_i - 1], n_agent_num[policy_i])
        players_id.append(players_id_list)

        action_space_list = [g.get_single_action_space(player_id) for player_id in players_id_list]
        actions_space.append(action_space_list)

    return players_id, actions_space


def get_joint_action_eval(game, multi_part_agent_ids, policy_list, actions_spaces, all_observes):
    if len(policy_list) != len(game.agent_nums):
        error = "模型个数%d与玩家个数%d维度不正确！" % (len(policy_list), len(game.agent_nums))
        raise Exception(error)

    # [[[0, 0, 0, 1]], [[0, 1, 0, 0]]]
    joint_action = []
    for policy_i in range(len(policy_list)):

        if game.obs_type[policy_i] not in obs_type:
            raise Exception("可选obs类型：%s" % str(obs_type))

        agents_id_list = multi_part_agent_ids[policy_i]

        action_space_list = actions_spaces[policy_i]
        function_name = 'm%d' % policy_i
        for i in range(len(agents_id_list)):
            agent_id = agents_id_list[i]
            a_obs = all_observes[agent_id]
            each = eval(function_name)(a_obs, action_space_list[i], game.is_act_continuous)
            joint_action.append(each)
    print(joint_action)
    return joint_action


def set_seed(g, env_name):
    if env_name.split("-")[0] in ['magent']:
        g.reset()
        seed = g.create_seed()
        g.set_seed(seed)


def render_game(g, fps=1):
    """
    This function is used to generate log for pygame rendering locally and render in time.
    The higher the fps, the faster the speed for rendering next step.
    only support gridgame:
    "gobang_1v1", "reversi_1v1", "snakes_1v1", "sokoban_2p", "snakes_3v3", "snakes_5p", "sokoban_1p", "cliffwalking"
    """

    import pygame
    pygame.init()
    screen = pygame.display.set_mode(g.grid.size)
    pygame.display.set_caption(g.game_name)
    clock = pygame.time.Clock()
    for i in range(len(policy_list)):
        if policy_list[i] not in get_valid_agents():
            raise Exception("agent {} not valid!".format(policy_list[i]))

        file_path = os.path.dirname(os.path.abspath(__file__)) + "/examples/algo/" + policy_list[i] + "/submission.py"
        if not os.path.exists(file_path):
            raise Exception("file {} not exist!".format(file_path))

        import_path = '.'.join(file_path.split('/')[-3:])[:-3]
        function_name = 'm%d' % i
        import_name = "my_controller"
        import_s = "from %s import %s as %s" % (import_path, import_name, function_name)
        print(import_s)
        exec(import_s, globals())

    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info = dict(game_name=env_type, n_player=g.n_player, board_height=g.board_height, board_width=g.board_width,
                     init_state=str(g.get_render_data(g.current_state)), init_info=str(g.init_info), start_time=st,
                     mode="window", render_info={"color": g.colors, "grid_unit": g.grid_unit, "fix": g.grid_unit_fix})

    all_observes = g.all_observes
    while not g.is_terminal():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        step = "step%d" % g.step_cnt
        print(step)
        game_info[step] = {}
        game_info[step]["time"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        joint_act = get_joint_action_eval(g, multi_part_agent_ids, policy_list, actions_space, all_observes)
        next_state, reward, done, info_before, info_after = g.step(joint_act)
        if info_before:
            game_info[step]["info_before"] = info_before
        game_info[step]["joint_action"] = str(joint_act)

        pygame.surfarray.blit_array(screen, g.render_board().transpose(1, 0, 2))
        pygame.display.flip()

        game_info[step]["state"] = str(g.get_render_data(g.current_state))
        game_info[step]["reward"] = str(reward)

        if info_after:
            game_info[step]["info_after"] = info_after

        clock.tick(fps)

    game_info["winner"] = g.check_win()
    game_info["winner_information"] = str(g.won)
    game_info["n_return"] = str(g.n_return)
    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info["end_time"] = ed


def run_game(g, env_name, multi_part_agent_ids, actions_spaces, policy_list, render_mode):
    """
    This function is used to generate log for Vue rendering. Saves .json file
    """
    from agents.random import submission as random_policy
    log_path = os.getcwd() + '/logs/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger = get_logger(log_path, g.game_name, json_file=render_mode)
    set_seed(g, env_name)

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    device = 'cpu'

    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    for i in range(len(policy_list)):
        if policy_list[i] not in get_valid_agents():
            raise Exception("agent {} not valid!".format(policy_list[i]))

        file_path = os.path.dirname(os.path.abspath(__file__)) + "/agents/" + policy_list[i] + "/submission.py"
        if not os.path.exists(file_path):
            raise Exception("file {} not exist!".format(file_path))

        import_path = '.'.join(file_path.split('/')[-3:])[:-3]
        function_name = 'm%d' % i
        import_name = "my_controller"
        import_s = "from %s import %s as %s" % (import_path, import_name, function_name)
        print(import_s)
        exec(import_s, globals())

    st = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info = {"game_name": env_name,
                 "n_player": g.n_player,
                 "board_height": g.board_height if hasattr(g, "board_height") else None,
                 "board_width": g.board_width if hasattr(g, "board_width") else None,
                 "init_info": g.init_info,
                 "start_time": st,
                 "mode": "terminal",
                 "seed": g.seed if hasattr(g, "seed") else None,
                 "map_size": g.map_size if hasattr(g, "map_size") else None}

    steps = []
    all_observes = g.all_observes # {'obs': {'agent_obs': (40, 40)array([[0., 0., 0., ... 0., 0.]]), 'id': 'team_1'}, 'controlled_player_index': 1}
    # [Box(-100, 200, shape=(1,)), Box(-30, 30, shape=(1,))]

    submission.rollouts.obs[0].copy_(torch.from_numpy(all_observes[1]['obs']['agent_obs'].reshape(-1, 40, 40)))
    submission.rollouts.obs[0].to(device)
    num_updates = int(
        100000)
    episode_rewards = deque(maxlen=10)
    for j in range(num_updates):
        g.reset()
        all_observes = g.all_observes # {'obs': {'agent_obs': (40, 40)array([[0., 0., 0., ... 0., 0.]]), 'id': 'team_1'}, 'controlled_player_index': 1}
        # [Box(-100, 200, shape=(1,)), Box(-30, 30, shape=(1,))]

        submission.rollouts.obs[0].copy_(torch.from_numpy(all_observes[1]['obs']['agent_obs'].reshape(-1, 40, 40)))
        submission.rollouts.obs[0].to(device)
        # decrease learning rate linearly
        start = time.time()
        ppo_utils.update_linear_schedule(
        submission.agent.optimizer, j, num_updates,
        5e-4)
        for step in range(1500):
            joint_act = []
            if hasattr(g, "env_core"):
                if hasattr(g.env_core, "render"):
                    g.env_core.render()
            with torch.no_grad():
                value, [action1, action2], [action_log_prob1, action_log_prob2], recurrent_hidden_states = submission.actor_critic.act(
                submission.rollouts.obs[step], submission.rollouts.recurrent_hidden_states[step],
                submission.rollouts.masks[step])
                random_actions = random_policy.my_controller(all_observes[0], actions_spaces[0][0], True)
            #joint_act = get_joint_action_eval(g, multi_part_agent_ids, policy_list, actions_spaces, all_observes)
            joint_act.append(random_actions)
            joint_act.append([action1.cpu().numpy(), action2.cpu().numpy()])
            temp_reward = 0
            if action1.cpu().numpy()[0] in [-100., 200.]:
                #print("large")
                temp_reward = -1

            # print(joint_act)
            all_observes, reward, done, info_before, info_after = g.step(joint_act)

            episode_rewards.append(reward[1])
            reward[1] += temp_reward
            #print(reward[1])
            masks = torch.FloatTensor(
                [0.0] if done else [1.0])
            bad_masks = torch.FloatTensor(
                [1.0]
                )
            submission.rollouts.insert(torch.from_numpy(all_observes[1]['obs']['agent_obs'].reshape(-1, 40, 40)), recurrent_hidden_states, torch.concat([action1, action2], dim=0),
                            torch.concat([action_log_prob1, action_log_prob2],  dim=0), value, torch.from_numpy(np.array(reward[1])), masks, bad_masks)
            if g.is_terminal():
                break
                g.reset()
                all_observes = g.all_observes # {'obs': {'agent_obs': (40, 40)array([[0., 0., 0., ... 0., 0.]]), 'id': 'team_1'}, 'controlled_player_index': 1}
    # [Box(-100, 200, shape=(1,)), Box(-30, 30, shape=(1,))]

                submission.rollouts.obs[0].copy_(torch.from_numpy(all_observes[1]['obs']['agent_obs'].reshape(-1, 40, 40)))
                submission.rollouts.obs[0].to(device)

        with torch.no_grad():
            next_value = submission.actor_critic.get_value(
                submission.rollouts.obs[-1], submission.rollouts.recurrent_hidden_states[-1],
                submission.rollouts.masks[-1]).detach()

        submission.rollouts.compute_returns(next_value, True, 0.99,
                                0.95, True)
        
        value_loss, action_loss, dist_entropy = submission.agent.update(submission.rollouts)
        print("Updating!! Step %d" % j)
        submission.rollouts.after_update()

        if (j % 100 == 0
            or j == num_updates - 1):
            save_path = os.path.join('./trained_models/', 'ppo')
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            #torch.save(pretrain_actor_critic.state_dict(), "./pretrain_actor_critic.pt")
            torch.save(
                submission.actor_critic.state_dict()
            , os.path.join(save_path, "olympics-integrated" + ".pt"))
        if j+1 % 500 == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * 1 * 1000
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))


    #while not g.is_terminal():
        # step = "step%d" % g.step_cnt
        # if g.step_cnt % 10 == 0:
        #     print(step)

        # if hasattr(g, "env_core"):
        #     if hasattr(g.env_core, "render"):
        #         g.env_core.render()
    #     info_dict = {"time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}
    #     joint_act = get_joint_action_eval(g, multi_part_agent_ids, policy_list, actions_spaces, all_observes)
    #     all_observes, reward, done, info_before, info_after = g.step(joint_act)
    #     if env_name.split("-")[0] in ["magent"]:
    #         info_dict["joint_action"] = g.decode(joint_act)
    #     if info_before:
    #         info_dict["info_before"] = info_before
    #     info_dict["reward"] = reward
    #     if info_after:
    #         info_dict["info_after"] = info_after
    #     steps.append(info_dict)

    game_info["steps"] = steps
    game_info["winner"] = g.check_win()
    game_info["winner_information"] = g.won
    game_info["n_return"] = g.n_return
    ed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    game_info["end_time"] = ed
    logs = json.dumps(game_info, ensure_ascii=False, cls=NpEncoder)
    logger.info(logs)
    print(game_info)


def get_valid_agents():
    dir_path = os.path.join(os.path.dirname(__file__), 'agents')
    return [f for f in os.listdir(dir_path) if f != "__pycache__"]


if __name__ == "__main__":

    env_type = "olympics-integrated"
    game = make(env_type, seed=None)

    render_mode = False

    render_in_time = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="random", help="random")
    parser.add_argument("--opponent", default="random", help="random")
    args = parser.parse_args()
    

    # policy_list = ["random"] * len(game.agent_nums)
    policy_list = [args.opponent, args.my_ai] #["random"] * len(game.agent_nums), here we control agent 2 (green agent)

    multi_part_agent_ids, actions_space = get_players_and_action_space_list(game)
    if render_in_time:
        render_game(game)
    else:
        run_game(game, env_type, multi_part_agent_ids, actions_space, policy_list, render_mode)