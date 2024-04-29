import os
import random
import time
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from mpi4py import MPI
import psutil
from copy import deepcopy as dc
from gymnasium_robotics.envs.training4.DomainRandomisation.randomisation import RandomisationModule
from Real4.Real_World_Interaction.Real_World_Interaction import RealEnv
import pickle
import csv

from agent import Agent
# from play import Play
# Assume friction_change.friction_change is a module you have that contains the change_friction logic
from Friction_Change.friction_change import FRICTION
from common.common import COMMON
from common.rotation import ROTATION

to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
MAX_EPOCHS = 50
MAX_CYCLES = 50
num_updates = 60
MAX_EPISODES = 2


class ActionUtility:
    def __init__(self, env):
        self.friction = FRICTION()
        self.action_functions = COMMON(env)
        self.rotation = ROTATION(env)

    @staticmethod
    def discretize_action_to_control_mode_E2E(action):
        # Your action discretization logic here
        action_norm = (action + 1) / 2
        if 1 / 6 > action_norm >= 0:
            control_mode = 0
            friction_state = 1  # left finger high friction
        elif 2 / 6 > action_norm >= 1 / 6:
            control_mode = 1
            friction_state = 1
        elif 3 / 6 > action_norm >= 2 / 6:
            control_mode = 2
            friction_state = -1
        elif 4 / 6 > action_norm >= 3 / 6:
            control_mode = 3
            friction_state = -1
        elif 5 / 6 > action_norm >= 4 / 6:
            control_mode = 4
            friction_state = 0
        else:
            assert 1 >= action_norm >= 5 / 6
            control_mode = 5
            friction_state = 0
        return friction_state, control_mode

    @staticmethod
    def discretize_action_to_control_mode(action):
        # Your action discretization logic here
        action_norm = (action + 1) / 2
        if 1 / 4 > action_norm >= 0:
            control_mode = 0
            friction_state = 1
        elif 2 / 4 > action_norm >= 1 / 4:
            control_mode = 1
            friction_state = 1
        elif 3 / 4 > action_norm >= 2 / 4:
            control_mode = 2
            friction_state = -1
        else:
            assert 1 > action_norm >= 3 / 4
            control_mode = 3
            friction_state = -1
        return friction_state, control_mode


    def friction_change(self, friction_state, env):
        friction_action_1 = [2, 0, True]
        friction_action_2 = [2, friction_state, True]
        # input("Press Enter to continue...")
        new_obs, rewards, terminated, _, infos = self.friction.change_friction_full_obs(np.array(friction_action_1),
                                                                                        env)
        if terminated:
            print("terminate at friction change to high")
            return new_obs, rewards["RL_IHM"], terminated, _, infos
        # input("press")
        new_obs, rewards, terminated, _, infos = self.friction.change_friction_full_obs(np.array(friction_action_2),
                                                                                        env)
        if terminated:
            print("terminate at friction change to low")
        return new_obs, rewards["RL_IHM"], terminated, _, infos

    def friction_change_to_high(self, env):
        friction_action_1 = [2, 0, True]
        new_obs, rewards, terminated, _, infos = self.friction.change_friction_full_obs(np.array(friction_action_1),
                                                                                        env)
        if terminated:
            print("terminate at friction change to high")
        return new_obs, rewards, terminated, _, infos


class EnvironmentSetup(ActionUtility):
    def __init__(self, env_name_: str, render=False, real=False, display=False):
        self.env_name = env_name_
        self.env = self.initialize_env(render)
        super().__init__(self.env)
        self.agent = self.initialize_agent()
        if real:
            self.real_env = RealEnv(self.env, display=display)

    def initialize_env(self, render):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["IN_MPI"] = "1"
        if render:
            env = gym.make(self.env_name, render_mode="human")
        else:
            env = gym.make(self.env_name)
        env.reset(seed=MPI.COMM_WORLD.Get_rank())
        random.seed(MPI.COMM_WORLD.Get_rank())
        np.random.seed(MPI.COMM_WORLD.Get_rank())
        torch.manual_seed(MPI.COMM_WORLD.Get_rank())
        return env

    def initialize_agent(self):
        # state_shape = self.env.observation_space.spaces["observation"].shape
        # n_actions = self.env.action_space.shape[0]
        # n_goals = self.env.observation_space.spaces["desired_goal"].shape[0]
        # action_bounds = [self.env.action_space.low[0], self.env.action_space.high[0]]

        test_env = gym.make("VariableFriction-v1")
        state_shape = test_env.observation_space.spaces["observation"].shape
        n_actions = test_env.action_space.shape[0]
        n_goals = test_env.observation_space.spaces["desired_goal"].shape[0]
        action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

        agent = Agent(
            n_states=state_shape,
            n_actions=n_actions,
            n_goals=n_goals,
            action_bounds=action_bounds,
            capacity=1232000,
            action_size=n_actions,
            batch_size=2048,
            actor_lr=1e-3,
            critic_lr=1e-3,
            gamma=0.98,
            tau=0.05,
            k_future=4,
            env=dc(self.env)
        )
        return agent


class TrainEvaluateAgent(EnvironmentSetup):
    def __init__(self, env_name_, render, real, display):
        super().__init__(env_name_, render=render, real=real, display=display)
        self.domain_randomise = RandomisationModule()
        self.r_dict = None

    def reset_environment(self):
        while True:
            env_dict = self.env.reset()[0]
            if np.mean(abs(env_dict["achieved_goal"] - env_dict["desired_goal"])) > 0.02:
                return env_dict

    def choose_action_with_filter(self, state, desired_goal, reward_dict, t, control_mode, train_mode):
        action, state_norm, desired_goal_norm = self.agent.choose_action(state, desired_goal, train_mode=train_mode)
        ''' Action feasibility filter '''
        # print(reward_dict)
        if reward_dict["pos_control_position"] <= 0.03 \
                or reward_dict["pos_control_position"] >= 1.65 \
                or reward_dict["torque_control_position"] >= 1.65 \
                or reward_dict["torque_control_position"] <= 0.03 \
                and t > 0:
            i = 0
            while True:
                i += 1
                if i > 10:
                    action[1] = np.random.uniform(-1, 1)
                friction_state, new_control_mode = self.discretize_action_to_control_mode(action[1])
                print("Check: ", new_control_mode, control_mode)
                if control_mode == new_control_mode \
                        or (control_mode == 0 and new_control_mode == 3) \
                        or (control_mode == 1 and new_control_mode == 2) \
                        or (control_mode == 2 and new_control_mode == 1) \
                        or (control_mode == 3 and new_control_mode == 0):
                    action, state_norm, desired_goal_norm = self.agent.choose_action(state, desired_goal, train_mode=train_mode)
                else:
                    return action, friction_state, new_control_mode, state_norm, desired_goal_norm
        friction_state, new_control_mode = self.discretize_action_to_control_mode(action[1])
        return action, friction_state, new_control_mode, state_norm, desired_goal_norm

    @staticmethod
    def extract_env_info(env_dict):
        return env_dict["observation"], env_dict["achieved_goal"], env_dict["desired_goal"]

    def run_episode(self, train_mode, randomisation=False, withRotation=False, withPause=False, real=False, display=False, reset=True, collect_demonstration=False, success_threshold=0.003, E2E=True):
        episode_dict = {
            "state": [],
            "action": [],
            "info": [],
            "achieved_goal": [],
            "desired_goal": [],
            "desired_goal_radi": [],
            "next_state": [],
            "next_achieved_goal": [],
            "reward": [],
            "terminals": []
        }
        friction_change_times = 0

        plt_dict = {
            "achieved_goal": []
        }
        per_success_rate = []
        episode_reward = 0
        print("Random: ", randomisation)

        if reset:
            if not real:
                env_dict = self.reset_environment()
                print("Reset everything")
            else:
                env_dict = self.reset_environment()
                env_dict_real = self.real_env.reset_robot()
        else:
            if real:
                _ = self.reset_environment()
                env_dict = self.real_env.get_obs_real()
            else:
                env_dict = self.env.get_new_goal()
                print(env_dict)

        state, achieved_goal, desired_goal = self.extract_env_info(env_dict)
        if randomisation:
            print("Randomisation: ", state[0], state[2])
            joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
            state[0] += joint_noise[0]
            state[2] += joint_noise[1]
            # print("Randomisation: ", state[0], state[2])

        ''' Pick up if terminated '''
        inAir = False
        last_friction = 0
        slide_success_real = 0
        if reset:
            if not real:
                env_dict, reward_dict = self.action_functions.pick_up(inAir)
                state, achieved_goal, desired_goal = self.extract_env_info(env_dict)
                self.r_dict = reward_dict
            else:
                # reward_dict = self.action_functions.pick_up(inAir)
                env_dict = self.real_env.pick_up_real(inAir)
                pos_idx = 0
                state, achieved_goal, desired_goal = self.extract_env_info(env_dict)

        ''' Step '''
        for t in range(20):
            if t == 0:
                action, state_norm, desired_goal_norm = self.agent.choose_action(state, desired_goal, train_mode=train_mode)
                friction_state, control_mode = self.discretize_action_to_control_mode(action[1])
            elif real:
                action, friction_state, control_mode, state_norm, desired_goal_norm = \
                    self.choose_action_with_filter_real(state,
                                                        desired_goal,
                                                        t,
                                                        control_mode,
                                                        pos_idx,
                                                        train_mode=train_mode)
            else:
                action, friction_state, control_mode, state_norm, desired_goal_norm = \
                    self.choose_action_with_filter(state,
                                                   desired_goal,
                                                   self.r_dict,
                                                   t,
                                                   control_mode,
                                                   train_mode=train_mode)

            # print("Action: ", action[0])
            # print("Friction change: ", friction_state)
            if not real:
                ''' Slide '''
                if friction_state != last_friction:
                    friction_change_times += 1
                    next_env_dict, r, terminated, _, info_ = self.friction_change(friction_state, self.env)
                    last_friction = friction_state
                    if terminated is False:
                        for _ in range(11):
                            next_env_dict, self.r_dict, terminated, _, info_ = self.env.step(action)
                            r = self.r_dict["RL_IHM"]
                            if terminated is True:
                                # input("Press to continue")
                                print("Terminated during the step")
                            if self.r_dict["pos_control_position"] <= 0.03 \
                                    or self.r_dict["torque_control_position"] >= 1.65 \
                                    or terminated is True:
                                break
                else:
                    for _ in range(11):
                        next_env_dict, self.r_dict, terminated, _, info_ = self.env.step(action)
                        r = self.r_dict["RL_IHM"]
                        if terminated is True:
                            # input("Press to continue")
                            print("Terminated during the step")
                        if self.r_dict["pos_control_position"] <= 0.03 \
                                or self.r_dict["torque_control_position"] >= 1.65 \
                                or terminated is True:
                            break
            else:
                ''' Slide on real robot '''
                if friction_state != last_friction:
                    friction_change_times += 1
                    friction_action_2 = [2, friction_state, True]
                    self.real_env.change_friction_real(friction_action_2)
                    last_friction = friction_state
                    for _ in range(11):
                        pos_idx = self.real_env.step_real(action)
                        next_env_dict = self.real_env.get_obs_real(display=display)
                        plt_dict["achieved_goal"].append(next_env_dict["object_pose"])
                        if next_env_dict["observation"][pos_idx*2] <= 0.03 \
                                or next_env_dict["observation"][(1 - pos_idx) * 2] >= 1.65:
                            break
                else:
                    for _ in range(11):
                        pos_idx = self.real_env.step_real(action)
                        next_env_dict = self.real_env.get_obs_real(display=display)
                        plt_dict["achieved_goal"].append(next_env_dict["object_pose"])
                        if next_env_dict["observation"][pos_idx * 2] <= 0.03 \
                                or next_env_dict["observation"][(1 - pos_idx) * 2] >= 1.65:
                            break

            '''Synchronize real pose to MuJoCo'''
            # if real:
            #     object_real_pose = next_env_dict['object_pose']
            #     self.env.step(object_real_pose)

            '''Plot IHM trajectory'''
            # if real:
            #     self.real_env.plot_track_real(display=False)
            #     plt.figure()
            #     for coord in plt_dict["achieved_goal"]:
            #         plt.scatter(coord[0], coord[1])
            if collect_demonstration:
                episode_dict["state"].append(state_norm[0].copy())
                episode_dict["action"].append(np.array(action.copy()))
                episode_dict["achieved_goal"].append(np.array(achieved_goal.copy()))
                # episode_dict["desired_goal"].append(np.array(desired_goal_norm[0].copy()))
                episode_dict['reward'].append(r.copy())
                episode_dict['terminals'].append(terminated)
            else:
                episode_dict["state"].append(state.copy())
                episode_dict["action"].append(action.copy())
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                episode_dict["desired_goal"].append(desired_goal.copy())

            next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(next_env_dict)
            if randomisation:
                print("Randomisation: ", next_state[0], next_state[2])
                joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
                next_state[0] += joint_noise[0]
                next_state[2] += joint_noise[1]
                # print("After Randomisation: ", next_state[0], next_state[2])
            state = next_state.copy()
            achieved_goal = next_achieved_goal.copy()
            desired_goal = next_desired_goal.copy()
            print("Radius: ", desired_goal, achieved_goal)

            if train_mode:
                ''' Train '''
                if terminated is True:
                    break
            elif not real:
                ''' Play '''
                per_success_rate.append(info_['is_success'])
                episode_reward += r
                if terminated is True:
                    print("Terminate: ", terminated)
                    break
                elif info_["is_success"] == 1:
                    print("Success: ", info_["is_success"])
                    break
            else:
                ''' Real '''
                if np.mean(np.array(abs(desired_goal - achieved_goal))) < success_threshold:
                    time.sleep(2)
                    print("SUCCESSSSSSSSSSSSS", desired_goal, achieved_goal)
                    slide_success_real = 1
                    per_success_rate.append(slide_success_real)
                    break
                else:
                    per_success_rate.append(slide_success_real)

        if real:
            pass
        elif collect_demonstration:
            last_achieved_goal = self.r_dict["goal_pose"]
            last_achieved_goal_radi = next_achieved_goal
            episode_dict["desired_goal"] = [last_achieved_goal for _ in episode_dict["achieved_goal"]]
            episode_dict["desired_goal_radi"] = [last_achieved_goal_radi for _ in episode_dict["achieved_goal"]]
            episode_dict["terminals"][-1] = True
            episode_dict["next_state"] = episode_dict["state"][1:]
            episode_dict["next_state"].append(next_state)
            episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
            episode_dict["next_achieved_goal"].append(next_achieved_goal)
        else:
            episode_dict["state"].append(state.copy())
            episode_dict["achieved_goal"].append(achieved_goal.copy())
            episode_dict["desired_goal"].append(desired_goal.copy())
            episode_dict["next_state"] = episode_dict["state"][1:]
            episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]

        slide_result = np.array(desired_goal - achieved_goal)
        print(f"------------------------------------{t}")
        # print(episode_dict)

        ''' Rotation '''
        if not E2E:
            if not real:
                if withRotation is True and info_["is_success"] == 1:
                    slide_success_real = info_["is_success"]
                    rotation_precision = 0.003
                    _, self.r_dict, _, _, _ = self.friction_change_to_high(self.env)
                    # print(r_dict)
                    # action[0], pos_idx = env.switch_ctrl_type(0)
                    rotation_action = np.array([self.r_dict['pos_control_position'], 0, False])
                    print("Start Rotation")
                    terminated, reward_dict = self.rotation.start_rotation(rotation_action, rotation_precision)
                    if terminated:
                        print("\033[91m| Terminated during rotation \033[0m")
                    # print("rotation complete")
                    # print(success, reward)
                    elif not terminated and reward_dict['current_goal_centre_distance'] < rotation_precision:
                        print("\033[92m| Rotation Achieved with reward of: ", reward_dict['current_goal_centre_distance'], "\033[0m")
                        print("\033[92m| SUCCESS \033[0m")
                        print("\033[92m--------------------------------------------------------------------\033[0m")
                        if withPause is True:
                            time.sleep(2)
                        # num_success += 1
                        # average_success_reward += reward
                    else:
                        print("\033[91m| Rotation Failed with reward of: ", reward_dict['current_goal_centre_distance'], "\033[0m")
                        print("\033[91m| FAILED \033[0m")
                        print("\033[92m--------------------------------------------------------------------\033[0m")
                        # if Test_WithPause is True and Train is False:
                        #     time.sleep(2)
                        # average_fail_reward += reward
                    # time.sleep(2)
                    # action[0], pos_idx = env.switch_ctrl_type_direct()
                    action, action_complete = self.rotation.reverse_rotate()
                    print("check action: ", action)
            else:
                if slide_success_real:
                    # print("Start rotation")
                    # print("Check: ", next_env_dict['object_pose'])
                    # print("Corners: ", next_env_dict["corners"])
                    # print("Goal: ", self.env.goal + [0, 0.12212, 0, 0, 0, 0, 0, 0, 0])
                    # input("Press to proceed")
                    # self.real_env.plot_track_real(display=True)
                    # plt.show()
                    friction_state = 0
                    friction_action_2 = [2, friction_state, True]
                    last_friction = friction_state
                    self.real_env.change_friction_real(friction_action_2)
                    _, _, pose_diff = self.real_env.start_rotation_real(self.env.goal.ravel().copy())
                    if withPause is True:
                        if real:
                            input("Press to continue")
                        else:
                            time.sleep(2)
                    if not reset:
                        self.real_env.reverse_rotation_real()
                else:
                    goal_gripper_frame = self.env.goal.ravel().copy() + [0, 0.12212, 0, 0, 0, 0, 0, 0, 0]
                    distance = np.linalg.norm(np.array(goal_gripper_frame[:2]) - np.array(next_env_dict['object_pose'][:2]))
                    achieved_goal_euler = self.real_env.convert_quat_to_euler(next_env_dict['object_pose'])
                    goal_euler = self.real_env.convert_quat_to_euler(goal_gripper_frame[:7])
                    angle_diff = achieved_goal_euler[5] + 90 - goal_euler[3] % 360
                    pose_diff = [distance, angle_diff]

                return episode_dict, per_success_rate, pose_diff, slide_result, friction_change_times

        # print(episode_dict)
        return episode_dict, per_success_rate, episode_reward, slide_result, friction_change_times

    def train(self, randomise):
        self.agent.load_weights()
        # Training logic here
        t_success_rate = []
        running_reward_list = []
        total_ac_loss = []
        total_cr_loss = []
        for epoch in range(MAX_EPOCHS):
            start_time = time.time()
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            for cycle in range(0, MAX_CYCLES):
                print("| Epoch: ", epoch, "| Cycle: ", cycle)
                mb = []
                cycle_actor_loss = 0
                cycle_critic_loss = 0
                for episode in range(MAX_EPISODES):
                    # print("Episode: ", episode)
                    episode_dict, _, _, _, _ = self.run_episode(train_mode=True, randomisation=randomise, E2E=True)
                    mb.append(dc(episode_dict))
                # print("store")
                self.agent.store(mb)
                # print("update")
                for n_update in range(num_updates):
                    actor_loss, critic_loss = self.agent.train()
                    cycle_actor_loss += actor_loss
                    cycle_critic_loss += critic_loss

                epoch_actor_loss += cycle_actor_loss / num_updates
                epoch_critic_loss += cycle_critic_loss / num_updates
                self.agent.update_networks()

                if (cycle + 1) % 10 == 0 and cycle != 0:
                    success_rate, running_reward, episode_reward = self.eval_agent(randomise=randomise)
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        ram = psutil.virtual_memory()
                        t_success_rate.append(success_rate)
                        running_reward_list.append(running_reward)
                        print(f"Epoch:{epoch}| "
                              f"Running_reward:{running_reward[-1]:.3f}| "
                              f"EP_reward:{episode_reward:.3f}| "
                              f"Memory_length:{len(self.agent.memory)}| "
                              f"Duration:{time.time() - start_time:.3f}| "
                              f"Actor_Loss:{actor_loss:.3f}| "
                              f"Critic_Loss:{critic_loss:.3f}| "
                              f"Success rate:{success_rate:.3f}| "
                              f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM"
                              )
                        self.agent.save_weights()
                        self.save_data(t_success_rate, running_reward_list)

                    if MPI.COMM_WORLD.Get_rank() == 0:
                        with SummaryWriter("logs") as writer:
                            for i, success_rate in enumerate(t_success_rate):
                                writer.add_scalar("Success_rate", success_rate, i)
                        plt.style.use('ggplot')
                        plt.figure()
                        plt.plot(np.arange(0, epoch * 5 + (cycle + 1) / 10), t_success_rate)
                        plt.title("Success rate")
                        plt.savefig("success_rate.png")
                        # plt.show()

            ram = psutil.virtual_memory()
            # success_rate, running_reward, episode_reward = eval_agent(env, agent)
            total_ac_loss.append(epoch_actor_loss)
            total_cr_loss.append(epoch_critic_loss)
            # if MPI.COMM_WORLD.Get_rank() == 0:
            #     t_success_rate.append(success_rate)
            #     print(f"Epoch:{epoch}| "
            #           f"Running_reward:{running_reward[-1]:.3f}| "
            #           f"EP_reward:{episode_reward:.3f}| "
            #           f"Memory_length:{len(agent.memory)}| "
            #           f"Duration:{time.time() - start_time:.3f}| "
            #           f"Actor_Loss:{actor_loss:.3f}| "
            #           f"Critic_Loss:{critic_loss:.3f}| "
            #           f"Success rate:{success_rate:.3f}| "
            #           f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")
            #     agent.save_weights()

        if MPI.COMM_WORLD.Get_rank() == 0:

            with SummaryWriter("logs") as writer:
                for i, success_rate in enumerate(t_success_rate):
                    writer.add_scalar("Success_rate", success_rate, i)

            plt.style.use('ggplot')
            plt.figure()
            plt.plot(np.arange(0, MAX_EPOCHS), t_success_rate)
            plt.title("Success rate")
            plt.savefig("success_rate.png")
            plt.show()
        pass

    def eval_agent(self, randomise):
        # Evaluation logic here
        total_success_rate = []
        running_r = []
        for ep in range(10):
            print("episode: ", ep)
            episode_dict, success, episode_reward = self.run_episode(train_mode=False,
                                                                                randomisation=randomise)
            total_success_rate.append(success[-1])
            if ep == 0:
                running_r.append(episode_reward)
            else:
                running_r.append(running_r[-1] * 0.99 + 0.01 * episode_reward)

        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate)
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r, episode_reward

    def play(self,
             num_episodes,
             randomise,
             withRotation=False,
             withPause=False,
             real=False,
             display=False,
             keep_reset=True,
             store_error=False,
             collect_demonstration=False,
             demontration_file_name=None,
             policy_path=None,
             success_threshold=0.003
             ):

        demonstration_dict = {
            'observations': [],
            'desired_goals': [],
            'desired_goals_radi': [],
            'actions': [],
            'next_observations': [],
            'rewards': [],
            'terminals': []
        }

        pose_error_list = []
        radi_error_list = []
        friction_change_times_list =[]
        if policy_path is not None:
            self.agent.load_weights_play(policy_path)
        else:
            self.agent.load_weights_play()
        self.agent.set_to_eval_mode()
        # self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        num_success = 0
        for ep in range(num_episodes):
            # print("episode: ", ep)
            if not keep_reset:
                ''' Only reset goal '''
                if ep == 0:
                    reset = True
                elif success[-1] == 0:
                    reset = True
                else:
                    reset = False
            else:
                ''' Reset everything '''
                reset = True
            episode_dict, success, episode_reward, slide_result, friction_change_times \
                = self.run_episode(train_mode=False,
                                   randomisation=randomise,
                                   withRotation=withRotation,
                                   withPause=withPause,
                                   real=real,
                                   display=display,
                                   reset=reset,
                                   collect_demonstration=collect_demonstration,
                                   success_threshold=success_threshold)
            if not real:
                if success[-1]:
                    num_success += 1
                    print(f"---- Episode {ep} SUCCESS ---- ")
                    # time.sleep(2)
                else:
                    print(f"---- Episode {ep} Failed ---- ")

                if (ep + 1) % 10 == 0:
                    print("Success rate: ", num_success / (ep + 1))
            else:
                pose_error_list.append(episode_reward)
                radi_error_list.append(slide_result)
                friction_change_times_list.append(friction_change_times)

            if collect_demonstration:
                if len(episode_dict['terminals']) > 10 or success[-1] == 0:
                    print("Not save")
                    continue
                else:
                    demonstration_dict['observations'].append(np.stack(episode_dict['state']))
                    demonstration_dict['next_observations'].append(np.stack(episode_dict['next_state']))
                    demonstration_dict['desired_goals'].append(np.stack(episode_dict['desired_goal']))
                    demonstration_dict['desired_goals_radi'].append(np.stack(episode_dict['desired_goal_radi']))
                    demonstration_dict['actions'].append(np.stack(episode_dict['action']))
                    demonstration_dict['rewards'].append(episode_dict['reward'])
                    demonstration_dict['terminals'].append(episode_dict['terminals'])

        if collect_demonstration:
            demonstration_dict['observations'] = np.vstack(demonstration_dict['observations']).astype(np.float32)
            demonstration_dict['next_observations'] = np.vstack(demonstration_dict['next_observations']).astype(np.float32)
            demonstration_dict['desired_goals'] = np.vstack(demonstration_dict['desired_goals']).astype(np.float32)
            demonstration_dict['desired_goals_radi'] = np.vstack(demonstration_dict['desired_goals_radi']).astype(np.float32)
            demonstration_dict['actions'] = np.vstack(demonstration_dict['actions']).astype(np.float32)
            demonstration_dict['rewards'] = np.hstack(demonstration_dict['rewards'])
            demonstration_dict['terminals'] = np.hstack(demonstration_dict['terminals'])
            # print(demonstration_dict)

            assert demontration_file_name is not None, f"File name is None, check: {demontration_file_name}"
            self.save_as_pickle(demonstration_dict, demontration_file_name)

        if store_error and real:
            # Save to CSV
            with open('play_evaluation.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Position Error', 'Orientation Error', 'Radi Error Left', 'Radi Error Right', 'Number of Friction Change'])  # Header
                for pose_error_, radi_error_, friction_change_times_ in zip(pose_error_list, radi_error_list, friction_change_times_list):
                    writer.writerow([pose_error_[0], pose_error_[1], radi_error_[0], radi_error_[1], friction_change_times_])
        # self.play_evaluation(np.array(pose_error_list), np.array(radi_error_list), np.array(friction_change_times_list))
        self.env.close()

    def save_as_pickle(self, data, filename, directory='demonstration'):
        os.makedirs(directory, exist_ok=True)
        full_path = os.path.join(directory, filename)

        try:
            with open(full_path, 'wb') as file:
                pickle.dump(data, file)
            print(f"Data successfully saved to {full_path}")
        except Exception as e:
            print(f"An error occurred while saving data: {e}")

    def modify_object_shape(self, size=0.015):
        object_idx_list = [16, 21]
        name_list = ['target', 'object']
        for i, obj_idx in enumerate(object_idx_list):
            self.env.model.geom_size[obj_idx] = [size, size, 0.05]
            contact_idx_1 = self.env._model_names._site_name2id[f"{name_list[i]}:corner1"]
            contact_idx_2 = self.env._model_names._site_name2id[f"{name_list[i]}:corner2"]
            contact_idx_3 = self.env._model_names._site_name2id[f"{name_list[i]}:corner3"]
            contact_idx_4 = self.env._model_names._site_name2id[f"{name_list[i]}:corner4"]

    def save_data(self, t_success_rate, running_reward_list):
        data = {
            't_success_rate': t_success_rate,
            'running_reward_list': running_reward_list
        }
        with open('data_check.pkl', 'wb') as f:
            pickle.dump(data, f)

    def choose_action_with_filter_real(self, state, desired_goal, t, control_mode, pos_idx, train_mode):
        action, state_norm, desired_goal_norm = self.agent.choose_action(state, desired_goal, train_mode=train_mode)
        ''' Action feasibility filter '''
        # print(reward_dict)
        if state[pos_idx*2] <= 0.03 \
                or state[pos_idx*2] >= 1.65 \
                or state[(1-pos_idx)*2] >= 1.65 \
                or state[(1-pos_idx)*2] <= 0.03 \
                and t > 0:
            i = 0
            while True:
                i += 1
                if i > 10:
                    action[1] = np.random.uniform(-1, 1)
                friction_state, new_control_mode = self.discretize_action_to_control_mode(action[1])
                if control_mode == new_control_mode \
                        or (control_mode == 0 and new_control_mode == 3) \
                        or (control_mode == 1 and new_control_mode == 2) \
                        or (control_mode == 2 and new_control_mode == 1) \
                        or (control_mode == 3 and new_control_mode == 0):
                    action, state_norm, desired_goal_norm = self.agent.choose_action(state, desired_goal, train_mode=train_mode)
                else:
                    return action, friction_state, new_control_mode, state_norm, desired_goal_norm
        friction_state, new_control_mode = self.discretize_action_to_control_mode(action[1])
        return action, friction_state, new_control_mode, state_norm, desired_goal_norm

    @staticmethod
    def play_evaluation(pose_error_list, radi_error_list, friction_change_times_list):
        if pose_error_list.size > 0:
            position_error_mean = np.mean(pose_error_list[:, 0])
            position_error_std = np.std(pose_error_list[:, 0])

            orientation_error_mean = np.mean(pose_error_list[:, 1])
            orientation_error_std = np.std(pose_error_list[:, 1])

            print(f"Position Error Mean: {position_error_mean:.3f}, Std: {position_error_std:.3f}")
            print(f"Orientation Error Mean: {orientation_error_mean:.3f}, Std: {orientation_error_std:.3f}")
        else:
            print("No position and orientation error data available.")

        if radi_error_list.size > 0:
            mean_radi_errors = np.mean(np.abs(radi_error_list), axis=1)
            radi_error_mean = np.mean(mean_radi_errors)
            radi_error_std = np.std(mean_radi_errors)
            print(f"Radi Error Mean: {radi_error_mean:.3f}, Std: {radi_error_std:.3f}")
        else:
            print("No radi error data available.")

        print(f"Number of friction change every episode, Mean: {np.mean(friction_change_times_list)}, "
              f"Std: {np.std(friction_change_times_list)}")


if __name__ == "__main__":
    ''' RL with framework, workable version in sim and real '''
    env_name = "VariableFriction-v5"
    real_ = False
    display_ = False
    trainer_evaluator = TrainEvaluateAgent(env_name,
                                           render=True,
                                           real=real_,
                                           display=display_)

    keep_reset_to_pick_up_pos = False
    demontration_file_name_ = "VFF-3"
    policy_path_ = '/Users/qiyangyan/Desktop/Training Files/Real4/Real4/pretrained_policy/VariableFriction.pth'  # 0.003
    # policy_path_ = '/Users/qiyangyan/Desktop/Training Files/Trained Policy/Training3_5mm_DR/VariableFriction.pth'  # 0.005
    trainer_evaluator.play(2000,
                           randomise=False,
                           withRotation=False,
                           withPause=False,
                           real=real_,
                           display=display_,
                           keep_reset=keep_reset_to_pick_up_pos,
                           store_error=True,
                           collect_demonstration=False,
                           demontration_file_name=demontration_file_name_,
                           policy_path=policy_path_,
                           success_threshold=0.003
                           )

    # trainer_evaluator.train(randomise=False)

    # while True:
    #     trainer_evaluator.reset_environment()
    #     input(f"Press, goal: {trainer_evaluator.real_env.convert_quat_to_euler(trainer_evaluator.env.goal[:7])}")