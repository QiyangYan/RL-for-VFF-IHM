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
# from Real4.Real4.Real_World_Interaction.Real_World_Interaction import RealEnv
from Real_World_Interaction.Real_World_Interaction import RealEnv
import pickle
import csv
import collections
import rotations

from agent import Agent
from agents.ql_diffusion import Diffusion_QL as Agent_diffusion
from agents.simple_nn import SimpleNN, EnhancedNN, TransformerTabNet
from utils.data_sampler import Data_Sampler
import argparse
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


def normalize(data, mean, std):
    # Ensure standard deviation is not zero to avoid division by zero error
    std = np.where(std == 0, 1, std)
    # print("normal", mean, std)
    return (data - mean) / std


class ActionUtility:
    def __init__(self, env):
        self.friction = FRICTION()
        self.action_functions = COMMON(env)
        self.rotation = ROTATION(env)

    @staticmethod
    def discretize_action_to_control_mode_E2E(action):
        """
        -1 ~ 1 maps to 0 ~ 1
        """
        # Your action discretization logic here
        # print("Action: ", action)
        action_norm = (action + 1) / 2
        # print(action_norm, action)
        if 1 / 6 > action_norm >= 0:
            print("| Slide up on right finger")
            control_mode = 0
            friction_state = 1  # left finger high friction
            pos_idx = 0
        elif 2 / 6 > action_norm >= 1 / 6:
            print("| Slide down on right finger")
            control_mode = 1
            friction_state = 1
            pos_idx = 1
        elif 3 / 6 > action_norm >= 2 / 6:
            print("| Slide up on left finger")
            control_mode = 2
            friction_state = -1
            pos_idx = 1
        elif 4 / 6 > action_norm >= 3 / 6:
            print("| Slide down on left finger")
            control_mode = 3
            friction_state = -1
            pos_idx = 0
        elif 5 / 6 > action_norm >= 4 / 6:
            print("| Rotate clockwise")
            control_mode = 4
            friction_state = 0
            pos_idx = 0
            # print("Rotate")
        else:
            assert 1 >= action_norm >= 5 / 6
            print("| Rotate anticlockwise")
            control_mode = 5
            friction_state = 0
            pos_idx = 1
            # print(pos_idx)
            # print("Rotate")
        return friction_state, control_mode, pos_idx

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
            assert 1 >= action_norm >= 3 / 4, f"Check: {action_norm}"
            control_mode = 3
            friction_state = -1
        return friction_state, control_mode

    def friction_change(self, friction_state, env):
        friction_action_1 = [2, 0, True]
        friction_action_2 = [2, friction_state, True]
        # input("Press Enter to continue...")
        new_obs, rewards, terminated, _, infos = self.friction.change_friction_full_obs(np.array(friction_action_1),
                                                                                        env)
        if terminated and friction_state == 0:
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

    def _goal_distance(self, goal_a, goal_b):
        ''' get pos difference and rotation difference
        left motor pos: 0.037012 -0.1845 0.002
        right motor pos: -0.037488 -0.1845 0.002
        '''
        assert goal_a.shape == goal_b.shape, f"Check: {goal_a.shape}, {goal_b.shape}"
        assert goal_a.shape[-1] == 7
        goal_a[2] = goal_b[2]

        d_pos = np.zeros_like(goal_a[..., 0])

        delta_pos = goal_a[..., :3] - goal_b[..., :3]
        d_pos = np.linalg.norm(delta_pos, axis=-1)

        quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

        euler_a = rotations.quat2euler(quat_a)
        euler_b = rotations.quat2euler(quat_b)
        if euler_a.ndim == 1:
            euler_a = euler_a[np.newaxis, :]  # Reshape 1D to 2D (1, 3)
        if euler_b.ndim == 1:
            euler_b = euler_b[np.newaxis, :]  # Reshape 1D to 2D (1, 3)
        euler_a[:, :2] = euler_b[:, :2]  # make the second and third term of euler angle the same
        quat_a = rotations.euler2quat(euler_a)
        quat_a = quat_a.reshape(quat_b.shape)

        # print(quat_a, quat_b)
        quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))  # q_diff = q1 * q2*
        angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1.0, 1.0))
        d_rot = angle_diff
        # assert d_pos.shape == d_rot.shape
        return d_pos, d_rot
        # return d_pos


class EnvironmentSetup(ActionUtility):
    def __init__(self, env_name_: str,
                 seed,
                 render=False,
                 real=False,
                 display=False,
                 diffusion=False,
                 local=False,
                 dataset=None,
                 output_dir=None,
                 model_idx=None,
                 layer_dim=256,
                 include_action_in_obs=False,
                 separate_policies=False,
                 terminate_indicator=False,
                 terminate_model_name=None,
                 termiante_save_path=None,
                 control_mode_indicator=False,
                 collect_demo=False,
                 seed_idx=None
                 ):

        self.seed_idx = seed_idx
        self.env_name = env_name_
        self.env = self.initialize_env(render, seed)
        self.separate_policies = separate_policies
        self.terminate_indicator = terminate_indicator
        self.control_mode_indicator = control_mode_indicator
        super().__init__(self.env)
        # print('3')
        if not separate_policies:
            print("Initialize agent here")
            self.agent = self.initialize_agent(env_name_,
                                               diffusion=diffusion,
                                               local=local,
                                               dataset=dataset,
                                               output_dir=output_dir,
                                               model_idx=model_idx,
                                               layer_dim=layer_dim,
                                               include_action_in_obs=include_action_in_obs,
                                               collect_demo=collect_demo
                                               )
        else:
            print("| Two agents")
            self.agent = self.initialize_agent(env_name_,
                                               diffusion=diffusion,
                                               local=local,
                                               dataset=dataset['dict1'],
                                               output_dir=output_dir[0],
                                               model_idx=model_idx[0],
                                               layer_dim=layer_dim,
                                               include_action_in_obs=include_action_in_obs,
                                               collect_demo=collect_demo,
                                               n_actions=1
                                               )

            self.agent_rotation = self.initialize_agent(env_name_,
                                                        diffusion=diffusion,
                                                        local=local,
                                                        dataset=dataset['dict2'],
                                                        output_dir=output_dir[1],
                                                        model_idx=model_idx[1],
                                                        layer_dim=layer_dim,
                                                        include_action_in_obs=include_action_in_obs,
                                                        collect_demo=collect_demo,
                                                        n_actions=2
                                                        )
            if self.terminate_indicator:
                print("| Terminate agents")
                # best_model_path = os.path.join(termiante_save_path, terminate_model_name, 'best_model.pth')
                # cube
                # best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_terminate_cube_for_real_WORK_DONTCHANGE/SimpleNN/best_model.pth'

                # cube cylinder
                # best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_terminate_cube_cylinder/SimpleNN/best_model.pth'

                # cube cylinder + No Obj Vel
                # best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_terminate_cube_cylinder_WORK_DONTCHANGE/SimpleNN/best_model.pth'

                # cube cylinder + No Obj Vel + No Joint Vel
                best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_terminate_cube_cylinder_WORK_DONTCHANGE_forReal/SimpleNN/best_model.pth'

                # three cylinder
                # best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_terminate_three_cylinder_WORK_DONTCHANGE/SimpleNN/best_model.pth'

                # cube 1 cm
                # best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_terminate_cube_1cm/SimpleNN/best_model.pth'

                # cube 2 cm
                # best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_terminate_cube_2cm/SimpleNN/best_model.pth'

                # mixed object
                # best_model_path = "/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_terminate_mixed_object/SimpleNN/best_model.pth"

                self.agent_terminate = torch.load(best_model_path)
            pass

            if self.control_mode_indicator:
                print("| Control mode agents")
                # cube
                # best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_operating_mode_cube_for_real_WORK_DONTCHANGE/SimpleNN/best_model.pth'

                # cube cylinder
                # best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_operating_mode_cube_cylinder/SimpleNN/best_model.pth'

                # cube cylinder + No Obj Vel
                # best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_operating_mode_cube_cylinder_WORK_DONTCHANGE/SimpleNN/best_model.pth'

                # cube cylinder + No Obj Vel + No Joint Vel
                best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_operating_mode_cube_cylinder_WORK_DONTCHANGE_forReal/SimpleNN/best_model.pth'

                # three cylinder
                # best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_operating_mode_three_cylinder_WORK_DONTCHANGE/SimpleNN/best_model.pth'

                # cube 1 cm
                # best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_operating_mode_cube_1cm/SimpleNN/best_model.pth'

                # cube 2 cm
                # best_model_path = '/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_operating_mode_cube_2cm/SimpleNN/best_model.pth'

                # mixed object
                # best_model_path = "/Users/qiyangyan/Downloads/diffusion_imitation-master/agents/models_operating_mode_mixed_object/SimpleNN/best_model.pth"

                self.agent_control_mode = torch.load(best_model_path)
                # print(self.agent_control_mode)

        if real:
            self.real_env = RealEnv(self.env, display=display)

    def initialize_env(self, render, seed):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["IN_MPI"] = "1"
        if render:
            env = gym.make(self.env_name, render_mode="human")
        else:
            env = gym.make(self.env_name)
        if seed:
            if self.seed_idx == None:
                self.seed_idx = MPI.COMM_WORLD.Get_rank()
            print("\033[94m | ------------------------------ \033[0m")
            print("\033[94m | ------------------------------ \033[0m")
            print(f"\033[94m | Seed: {self.seed_idx} \033[0m")
            print("\033[94m | ------------------------------ \033[0m")
            print("\033[94m | ------------------------------ \033[0m")
            env.reset(seed=self.seed_idx)
            random.seed(self.seed_idx)
            np.random.seed(self.seed_idx)
            torch.manual_seed(self.seed_idx)
        else:
            env.reset()
        return env

    def initialize_agent(self,
                         env_name,
                         diffusion=False,
                         local=False,
                         dataset=None,
                         output_dir=None,
                         model_idx=None,
                         layer_dim=256,
                         include_action_in_obs=False,
                         collect_demo=False,
                         n_actions=None
                         ):

        # state_shape = self.env.observation_space.spaces["observation"].shape
        # n_actions = self.env.action_space.shape[0]
        # n_goals = self.env.observation_space.spaces["desired_goal"].shape[0]
        # action_bounds = [self.env.action_space.low[0], self.env.action_space.high[0]]

        if env_name == "VariableFriction-v6" or (env_name == "VariableFriction-v7" and collect_demo == False):
            print(f"Initialize with environment: {env_name}")
            test_env = gym.make(env_name)
        else:
            print("Initialize with environment: VariableFriction-v5")
            test_env = gym.make("VariableFriction-v5")
        state_shape = test_env.observation_space.spaces["observation"].shape[0]
        if n_actions is None:
            n_actions = test_env.action_space.shape[0]
        n_goals = test_env.observation_space.spaces["desired_goal"].shape[0]
        action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]

        print("-"*50)
        print("State shape: ", state_shape)
        print(f"Action bounds: {action_bounds}")
        print(f"Action: {test_env.action_space.shape}")
        print(n_actions)
        print("-" * 50)

        if not diffusion:
            state_shape = test_env.observation_space.spaces["observation"].shape
            if not local:  # train on local computer with smaller batch size
                agent = Agent(
                    n_states=state_shape,
                    n_actions=n_actions,
                    n_goals=n_goals,
                    action_bounds=action_bounds,
                    capacity=1232000,
                    action_size=n_actions,
                    batch_size=1024,
                    actor_lr=1e-3,
                    critic_lr=1e-3,
                    gamma=0.98,
                    tau=0.05,
                    k_future=4,
                    local=False,
                    env=dc(self.env)
                )
            else:
                agent = Agent(
                    n_states=state_shape,
                    n_actions=n_actions,
                    n_goals=n_goals,
                    action_bounds=action_bounds,
                    capacity=7e+5 // 50,
                    action_size=n_actions,
                    batch_size=256,
                    actor_lr=1e-3,
                    critic_lr=1e-3,
                    gamma=0.98,
                    tau=0.05,
                    k_future=4,
                    local=True,
                    env=dc(self.env)
                )
        else:
            device = f"cuda:{0}" if torch.cuda.is_available() else "cpu"
            n_goals = dataset['desired_goals'].shape[1]

            if include_action_in_obs:
                state_dim_ = state_shape + n_actions + n_goals
            else:
                state_dim_ = state_shape + n_goals

            print("Check: ", state_dim_, state_shape, n_actions)
            agent = Agent_diffusion(
                state_dim=state_dim_,
                action_dim=n_actions,
                max_action=1,
                device=device,
                discount=0.99,
                tau=0.005,
                max_q_backup=False,
                eta=0.,  # BC only
                n_timesteps=5,
                lr=3e-4,
                lr_decay=True,
                lr_maxt=1000,
                grad_norm=1.0,
                layer_dim=layer_dim,
                include_action_in_obs=include_action_in_obs,
            )

            print(f"Load Model: {model_idx}")
            agent.load_model(output_dir, model_idx)
            agent.model.eval()
            agent.actor.eval()

            obs = dataset['observations'][1]
            goal = dataset['desired_goals'][1]
            true_action = dataset['actions'][1]

            # print(dataset['observations'][1])

            state = np.concatenate([goal, obs])
            # obs_norm = normalize(obs, agent.obs_mean, agent.obs_std)
            # goal_norm = normalize(goal, agent.goal_mean, agent.goal_std)
            # action = agent.sample_action(np.concatenate([state, np.zeros(2)]))
            if include_action_in_obs:
                action = agent.sample_action(np.concatenate([state, np.zeros(2)]))
            else:
                action = agent.sample_action(state)

            print(action, true_action)

        # print('5')

        return agent


def remove_useless_data(state, desired_goal, real):
    if state[1] > 1e-03:
        state[1] = 1
    else:
        state[1] = 0

    if state[3] > 1e-03:
        state[3] = 1
    else:
        state[3] = 0

    state[4] = 0
    state[5] = 0
    state[6] = 0
    state[7] = 0

    state[10] = 0

    state[12] = 0
    state[13] = 0

    if real is True:
        state[11] *= -1
        state[14] *= -1

    desired_goal[4] = 0
    desired_goal[5] = 0
    desired_goal[2] = 0


    if len(desired_goal) == 9:
        pass
    else:
        desired_goal = desired_goal[:-2]

    return state, desired_goal


class TrainEvaluateAgent(EnvironmentSetup):
    def __init__(self,
                 env_name_,
                 render,
                 real,
                 display,
                 diffusion=False,
                 local=False,
                 seed=True,
                 dataset=None,
                 output_dir=None,
                 model_idx=None,
                 layer_dim=256,
                 include_action_in_obs=False,
                 separate_policies=False,
                 terminate_indicator=False,
                 terminate_model_name=None,
                 termiante_save_path=None,
                 control_mode_indicator=False,
                 collect_demo=False,
                 seed_idx=None
                 ):
        super().__init__(env_name_,
                         render=render,
                         real=real,
                         display=display,
                         diffusion=diffusion,
                         local=local,
                         seed=seed,
                         dataset=dataset,
                         output_dir=output_dir,
                         model_idx=model_idx,
                         layer_dim=layer_dim,
                         include_action_in_obs=include_action_in_obs,
                         separate_policies=separate_policies,
                         terminate_indicator=terminate_indicator,
                         terminate_model_name=terminate_model_name,
                         termiante_save_path=termiante_save_path,
                         control_mode_indicator=control_mode_indicator,
                         collect_demo=collect_demo,
                         seed_idx=seed_idx
                         )
        self.include_action_in_obs = include_action_in_obs
        self.domain_randomise = RandomisationModule()
        self.r_dict = {}
        self.diffusion = diffusion

    def reset_environment(self, E2E=False, demo_collect_randomisation=False, repeat_same_goal=False):
        reset_specify = None
        if demo_collect_randomisation:
            reset_specify = {
                'repeat_same_goal': repeat_same_goal,
            }
        if E2E:
            env_dict = self.env.reset(options=reset_specify)[0]
            return env_dict
        else:
            while True:
                env_dict = self.env.reset(options=reset_specify)[0]
                if np.mean(abs(env_dict["achieved_goal"] - env_dict["desired_goal"])) > 0.02:
                    return env_dict

    def reset(self, real, reset, inAir=False, E2E=False, demo_collect_randomisation=False, repeat_same_goal=False):
        reward_dict = None
        if reset:
            if real:
                print("Reset everything | Real")
                _ = self.reset_environment(E2E=E2E, demo_collect_randomisation=demo_collect_randomisation,
                                           repeat_same_goal=repeat_same_goal)
                _ = self.real_env.reset_robot()
                env_dict = self.real_env.pick_up_real(inAir)
                reward_dict = env_dict.copy()
            else:
                print("Reset everything")
                _ = self.reset_environment(E2E=E2E, demo_collect_randomisation=demo_collect_randomisation,
                                           repeat_same_goal=repeat_same_goal)
                env_dict, reward_dict = self.action_functions.pick_up(inAir)
                self.r_dict = reward_dict
        else:
            if real:
                print("Reset goal | Real")
                _ = self.reset_environment(E2E=E2E, demo_collect_randomisation=demo_collect_randomisation,
                                           repeat_same_goal=repeat_same_goal)
                env_dict = self.real_env.get_obs_real()
            else:
                print("Reset goal")
                env_dict = self.env.get_new_goal()
        return env_dict, reward_dict

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
                # print("Check: ", new_control_mode, control_mode)
                if control_mode == new_control_mode \
                        or (control_mode == 0 and new_control_mode == 3) \
                        or (control_mode == 1 and new_control_mode == 2) \
                        or (control_mode == 2 and new_control_mode == 1) \
                        or (control_mode == 3 and new_control_mode == 0):
                    action, state_norm, desired_goal_norm = self.agent.choose_action(state, desired_goal,
                                                                                     train_mode=train_mode)
                else:
                    return action, friction_state, new_control_mode, state_norm, desired_goal_norm
        friction_state, new_control_mode = self.discretize_action_to_control_mode(action[1])
        return action, friction_state, new_control_mode, state_norm, desired_goal_norm

    @staticmethod
    def extract_env_info(env_dict, real=False):
        if real:
            return env_dict["observation"], env_dict["achieved_goal"], env_dict["desired_goal"], env_dict["demo_obs"], env_dict["demo_achieved_goal"]
        else:
            return env_dict["observation"], env_dict["achieved_goal"], env_dict["desired_goal"]

    def run_episode(self, train_mode, randomisation=False, withRotation=False, withPause=False, real=False,
                    display=False, reset=True, collect_demonstration=False, success_threshold=0.003):
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
            "terminals": [],
            'trajectory': []
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
        control_mode = None
        pos_idx = None
        terminated = None
        r = None
        if reset:
            if not real:
                env_dict, reward_dict = self.action_functions.pick_up(inAir)
                state, achieved_goal, desired_goal = self.extract_env_info(env_dict)
                self.r_dict = reward_dict
                if randomisation:
                    print("Randomisation: ", state[0], state[2])
                    joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
                    state[0] += joint_noise[0]
                    state[2] += joint_noise[1]
            else:
                # reward_dict = self.action_functions.pick_up(inAir)
                env_dict = self.real_env.pick_up_real(inAir)
                pos_idx = 0
                state, achieved_goal, desired_goal = self.extract_env_info(env_dict)
                if randomisation:
                    print("Randomisation: ", state[0], state[2])
                    joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
                    state[0] += joint_noise[0]
                    state[2] += joint_noise[1]

        ''' Step '''
        for t in range(20):
            if self.diffusion:
                ''' Since the RL policy was trained with those information '''
                if self.env_name == 'VariableFriction-v6':  # TODO: It was v6
                    # TODO: uncomment below for tasks:
                    #  1. for informative observation space
                    #  2. RL-aligned observation space

                    ' -1- '
                    # desired_goal_policyInput = desired_goal
                    # state_policyInput = state

                    ' -2- '
                    # desired_goal_policyInput = desired_goal
                    # state_policyInput = np.concatenate([state[:8], achieved_goal[7:9]])

                    ' -3- '
                    desired_goal_policyInput = desired_goal[:-2]
                    state_policyInput = state
                else:
                    desired_goal_policyInput = self.env.goal.ravel()[:7]
                    state_policyInput = state
            else:
                if self.env_name == 'VariableFriction-v7':
                    # desired_goal_policyInput = desired_goal[7:9]
                    # print(self.r_dict['desired_goal_contact_point_radi'])
                    desired_goal_policyInput = self.r_dict['desired_goal_contact_point_radi']
                    state_policyInput = np.concatenate([state[:8], self.r_dict['achieved_goal_contact_point_radi']])
                else:
                    desired_goal_policyInput = desired_goal
                    state_policyInput = state
            # print(state_policyInput)
            if t == 0:
                action, state_norm, desired_goal_norm = self.agent.choose_action(state_policyInput,
                                                                                 desired_goal_policyInput,
                                                                                 train_mode=train_mode)
                friction_state, control_mode = self.discretize_action_to_control_mode(action[1])
            elif real:
                action, friction_state, control_mode, state_norm, desired_goal_norm = \
                    self.choose_action_with_filter_real(state_policyInput,
                                                        desired_goal_policyInput,
                                                        t,
                                                        control_mode,
                                                        pos_idx,
                                                        train_mode=train_mode)
            else:
                action, friction_state, control_mode, state_norm, desired_goal_norm = \
                    self.choose_action_with_filter(state_policyInput,
                                                   desired_goal_policyInput,
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
                        if next_env_dict["observation"][pos_idx * 2] <= 0.03 \
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

            # print(action, control_mode)
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

            '''Store trajectories'''
            if collect_demonstration:
                episode_dict["state"].append(np.array(state).copy())  # original obs contains radi info, don't want that
                episode_dict["action"].append(np.array(action.copy()))
                episode_dict["achieved_goal"].append(np.array(achieved_goal.copy()))
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
            # print("Radius: ", desired_goal, achieved_goal)

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
                    # time.sleep(2)
                    print("SUCCESSSSSSSSSSSSS", desired_goal, achieved_goal)
                    slide_success_real = 1
                    per_success_rate.append(slide_success_real)
                    break
                else:
                    per_success_rate.append(slide_success_real)

        if real:
            pass
        elif collect_demonstration:
            if self.env_name == 'VariableFriction-v5':
                last_achieved_goal = self.r_dict["goal_pose"]
                last_achieved_goal_radi = next_achieved_goal
                # episode_dict["desired_goal_radi"] = [last_achieved_goal_radi for _ in episode_dict["achieved_goal"]]
            else:
                assert self.env_name == 'VariableFriction-v7', f"Current env is: {env_name}"
                assert len(next_achieved_goal) == 11, f"Check: {next_achieved_goal}"
                last_achieved_goal = next_achieved_goal
                episode_dict["next_state"].append(
                    next_state)  # the trained normalizer doesn't match the complete state dimension
            episode_dict["desired_goal"] = [last_achieved_goal for _ in episode_dict["achieved_goal"]]
            episode_dict["terminals"][-1] = True
            episode_dict["next_state"] = episode_dict["state"][1:]
            episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
            episode_dict["next_achieved_goal"].append(next_achieved_goal)
            if self.env_name == 'VariableFriction-v5':
                episode_dict["next_state"].append(self.agent.state_normalizer.normalize(next_state))
            else:
                episode_dict["next_state"].append(
                    next_state)  # the trained normalizer doesn't match the complete state dimension
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
        if not real:
            # if withRotation is True and info_["is_success"] == 1:
            #     slide_success_real = info_["is_success"]
            #     rotation_precision = 0.003
            #     next_env_dict, self.r_dict, _, _, _ = self.friction_change_to_high(self.env)
            #     # print(r_dict)
            #     # action[0], pos_idx = env.switch_ctrl_type(0)
            #     rotation_action = np.array([self.r_dict['pos_control_position'], 0, False])
            #     print("Start Rotation")
            #     print(self.env.pos_idx)
            #     terminated, reward_dict, _, _ = self.rotation.start_rotation(rotation_action, rotation_precision, episode_dict)
            #     if terminated:
            #         print("\033[91m| Terminated during rotation \033[0m")
            #     # print("rotation complete")
            #     # print(success, reward)
            #     elif not terminated and reward_dict['current_goal_centre_distance'] < rotation_precision:
            #         print("\033[92m| Rotation Achieved with reward of: ", reward_dict['current_goal_centre_distance'],
            #               "\033[0m")
            #         print("\033[92m| SUCCESS \033[0m")
            #         print("\033[92m--------------------------------------------------------------------\033[0m")
            #         if withPause is True:
            #             time.sleep(2)
            #         # num_success += 1
            #         # average_success_reward += reward
            #     else:
            #         print("\033[91m| Rotation Failed with reward of: ", reward_dict['current_goal_centre_distance'],
            #               "\033[0m")
            #         print("\033[91m| FAILED \033[0m")
            #         print("\033[92m--------------------------------------------------------------------\033[0m")
            #         # if Test_WithPause is True and Train is False:
            #         #     time.sleep(2)
            #         # average_fail_reward += reward
            #     # time.sleep(2)
            #     # action[0], pos_idx = env.switch_ctrl_type_direct()
            #     action, action_complete = self.rotation.reverse_rotate()
            #     print("check action: ", action)
            ''' Big step '''
            # action = np.zeros(2)
            # if withRotation is True:
            #     rotation_precision = 0.003
            #     _, self.r_dict, _, _, _ = self.friction_change_to_high(self.env)
            #     rotation_action = np.array([self.r_dict['pos_control_position'], 0, False])
            #     terminated, self.r_dict, env_dict, episode_dict = self.rotation.start_rotation(
            #         rotation_action.copy(), rotation_precision, episode_dict)
            #     next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(env_dict)
            #     # TODO: Modify the rad range to policy range,
            #     #  this is the range with unit rad instead of policy action as other steps,
            #     #  remember to modify the set_action for: VFF-bigSteps
            #     action[0] = abs(rotation_action[0] - self.r_dict["pos_control_position"])
            #     action[1] = 5 / 6 - 1 / 12 + 1 / 6 * self.env.pos_idx
            #
            #     episode_dict["state"].append(
            #         np.array(state.copy()))  # original obs contains radi info, don't want that
            #     episode_dict["achieved_goal"].append(np.array(achieved_goal.copy()))
            #     episode_dict['terminals'].append(terminated)
            #     episode_dict["action"].append(np.array(action.copy()))
            #     # print('Check_2: ', achieved_goal)
            #
            #     if terminated:
            #         print("\033[91m| Terminated during rotation \033[0m")
            #     elif not terminated and self.r_dict['current_goal_centre_distance'] < rotation_precision:
            #         print("\033[92m| Rotation Achieved with reward of: ",
            #               self.r_dict['current_goal_centre_distance'], "\033[0m")
            #         print("\033[92m| SUCCESS \033[0m")
            #         print("\033[92m--------------------------------------------------------------------\033[0m")
            #     else:
            #         print("\033[91m| Rotation Failed with reward of: ", self.r_dict['current_goal_centre_distance'],
            #               "\033[0m")
            #         print("\033[91m| FAILED \033[0m")
            #         print("\033[92m--------------------------------------------------------------------\033[0m")
            # action, action_complete = self.rotation.reverse_rotate()
            # print("check action: ", action)
            action = np.zeros(2)
            if withRotation is True and info_["is_success"] == 1:
                rotation_precision = 0.003
                _, self.r_dict, _, _, _ = self.friction_change_to_high(self.env)
                # print(self.env.pos_idx)
                rotation_action = np.array([self.r_dict['pos_control_position'], 0, False])
                # print(rotation_action)
                terminated, self.r_dict, env_dict, episode_dict = self.rotation.start_rotation(
                    rotation_action.copy(), rotation_precision, episode_dict)
                next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(env_dict)
                # TODO: Modify the rad range to policy range,
                #  this is the range with unit rad instead of policy action as other steps,
                #  remember to modify the set_action for: VFF-bigSteps
                action[0] = abs(rotation_action[0] - self.r_dict["pos_control_position"])
                action[1] = 5 / 6 - 1 / 12 + 1 / 6 * self.env.pos_idx
                action[1] = action[1] * 2 - 1

                episode_dict["state"].append(
                    np.array(state.copy()))  # original obs contains radi info, don't want that
                episode_dict["achieved_goal"].append(np.array(achieved_goal.copy()))
                episode_dict['terminals'].append(terminated)
                episode_dict["action"].append(np.array(action.copy()))
                # print('Check_2: ', achieved_goal)

                if terminated:
                    print("\033[91m| Terminated during rotation \033[0m")
                elif not terminated and self.r_dict['current_goal_centre_distance'] < rotation_precision:
                    print("\033[92m| Rotation Achieved with reward of: ",
                          self.r_dict['current_goal_centre_distance'], "\033[0m")
                    print("\033[92m| SUCCESS \033[0m")
                    print("\033[92m--------------------------------------------------------------------\033[0m")
                else:
                    print("\033[91m| Rotation Failed with reward of: ", self.r_dict['current_goal_centre_distance'],
                          "\033[0m")
                    print("\033[91m| FAILED \033[0m")
                    print("\033[92m--------------------------------------------------------------------\033[0m")
        else:
            if slide_success_real:
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

                episode_dict["state"].append(
                    np.array(state.copy()))  # original obs contains radi info, don't want that
                episode_dict["achieved_goal"].append(np.array(achieved_goal.copy()))
                episode_dict['terminals'].append(terminated)
                episode_dict["action"].append(np.array(action.copy()))
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

    def run_episode_E2E_test_distance_metric(self, train_mode):
        return NotImplementedError

    def train(self, randomise, E2E=False, test_distance_metric=False):
        # self.agent.load_weights()
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
                    if test_distance_metric is True:
                        episode_dict, _, _, _ = self.run_episode_E2E_test_distance_metric(train_mode=True)
                    elif E2E is True:
                        episode_dict, _, _, _, _, _ = self.run_episode_E2E(train_mode=True, randomisation=randomise)
                    else:
                        episode_dict, _, _, _, _ = self.run_episode(train_mode=True, randomisation=randomise, E2E=E2E)
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
                    if test_distance_metric is True:
                        success_rate, running_reward, episode_reward = self.eval_agent_E2E_test_distance_metric(
                            randomise=randomise)
                    elif E2E:
                        success_rate, running_reward, episode_reward = self.eval_agent_E2E(randomise=randomise)
                    else:
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

                        with SummaryWriter("logs") as writer:
                            for i, success_rate in enumerate(t_success_rate):
                                writer.add_scalar("Success_rate", success_rate, i)
                        plt.style.use('ggplot')
                        plt.figure()
                        plt.plot(np.arange(0, epoch * 5 + (cycle + 1) / 10), t_success_rate)
                        plt.title("Success rate")
                        plt.savefig("success_rate.png")
                        # plt.show()

            total_ac_loss.append(epoch_actor_loss)
            total_cr_loss.append(epoch_critic_loss)
            # ram = psutil.virtual_memory()
            # success_rate, running_reward, episode_reward = eval_agent(env, agent)
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

    def eval_agent_E2E(self, randomise):
        # Evaluation logic here
        total_success_rate = []
        running_r = []
        for ep in range(10):
            print("episode: ", ep)
            episode_dict, success, episode_reward, _, _, slide_success = self.run_episode_E2E(train_mode=False,
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

    def eval_agent_E2E_test_distance_metric(self, randomise):
        # Evaluation logic here
        total_success_rate = []
        running_r = []
        for ep in range(10):
            print("episode: ", ep)
            episode_dict, success, episode_reward, _ = self.run_episode_E2E_test_distance_metric(train_mode=False)

            total_success_rate.append(success[-1])
            if ep == 0:
                running_r.append(episode_reward)
            else:
                running_r.append(running_r[-1] * 0.99 + 0.01 * episode_reward)

        # print("Check 4")
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate)
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        # print("Check 5")
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
             success_threshold=0.003,
             test_distance_metric=False
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
        orientation_error_list = []
        friction_change_times_list = []
        slide_result = None
        if policy_path is not None:
            self.agent.load_weights_play(policy_path)
        else:
            self.agent.load_weights_play()
        self.agent.set_to_eval_mode()
        # self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        num_success = 0
        num_success_pos_rot = 0
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

            if test_distance_metric:
                episode_dict, success, episode_reward, friction_change_times = self.run_episode_E2E_test_distance_metric(
                    train_mode=True)
            else:
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
                    pos_error, orientation_error = self._goal_distance(np.array(self.env.goal[:7]),
                                                                       np.array(self.env.data.qpos[5:12]))
                    pose_error_list.append(pos_error)
                    orientation_error_list.append(orientation_error)
                    # time.sleep(2)

                    # BLUE = "\033[94m"
                    # YELLOW = "\033[93m"
                    # ENDC = "\033[0m"
                    # print(pos_error, orientation_error)
                    # if abs(pos_error) < self.env.d_threshold and abs(orientation_error) < self.env.angle_threshold:
                    #     num_success_pos_rot += 1
                    #     print(f"{BLUE}{'-' * 50}{ENDC}")
                    #     print(f"{BLUE}Success{ENDC}")
                    #     print(f"{BLUE}{'-' * 50}{ENDC}")
                    # else:
                    #     print(f"{YELLOW}{'-' * 50}{ENDC}")
                    #     print(f"{YELLOW}Failed{ENDC}")
                    #     print(f"{YELLOW}{'-' * 50}{ENDC}")
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
                    # if self.env_name == 'VariableFriction-v5':
                    #     demonstration_dict['desired_goals_radi'].append(np.stack(episode_dict['desired_goal_radi']))
                    demonstration_dict['observations'].append(np.stack(episode_dict['state']))
                    # demonstration_dict['next_observations'].append(np.stack(episode_dict['next_state']))
                    demonstration_dict['desired_goals'].append(np.stack(episode_dict['desired_goal']))
                    demonstration_dict['actions'].append(np.stack(episode_dict['action']))
                    demonstration_dict['terminals'].append(episode_dict['terminals'])

        if collect_demonstration:
            # if self.env_name == 'VariableFriction-v5':
            #     demonstration_dict['desired_goals_radi'] = np.vstack(demonstration_dict['desired_goals_radi']).astype(
            #         np.float32)
            demonstration_dict['observations'] = np.vstack(demonstration_dict['observations']).astype(np.float32)
            # demonstration_dict['next_observations'] = np.vstack(demonstration_dict['next_observations']).astype(np.float32)
            demonstration_dict['desired_goals'] = np.vstack(demonstration_dict['desired_goals']).astype(np.float32)
            demonstration_dict['actions'] = np.vstack(demonstration_dict['actions']).astype(np.float32)
            demonstration_dict['terminals'] = np.hstack(demonstration_dict['terminals'])
            # print(demonstration_dict)

            assert demontration_file_name is not None, f"File name is None, check: {demontration_file_name}"
            self.save_as_pickle(demonstration_dict, demontration_file_name)

        if store_error and real:
            # Save to CSV
            with open('play_evaluation.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Position Error', 'Orientation Error', 'Radi Error Left', 'Radi Error Right',
                                 'Number of Friction Change'])  # Header
                for pose_error_, radi_error_, friction_change_times_ in zip(pose_error_list, radi_error_list,
                                                                            friction_change_times_list):
                    writer.writerow(
                        [pose_error_[0], pose_error_[1], radi_error_[0], radi_error_[1], friction_change_times_])
        # self.play_evaluation(np.array(pose_error_list), np.array(radi_error_list), np.array(friction_change_times_list))
        self.env.close()

        if not real:
            print(pose_error_list, orientation_error_list)
            print("| Success rate: ", num_success_pos_rot / (ep + 1.0))
            print("| Success threshold: ", self.env.d_threshold)
            print(
                f"| Pos error statics: mean = {np.mean(np.array(pose_error_list))}, std = {np.std(np.array(pose_error_list))}")
            print(
                f"| Orientation error statics: mean = {np.mean(np.array(orientation_error_list))}, std = {np.std(np.array(orientation_error_list))}")

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
        if state[pos_idx * 2] <= 0.03 \
                or state[pos_idx * 2] >= 1.65 \
                or state[(1 - pos_idx) * 2] >= 1.65 \
                or state[(1 - pos_idx) * 2] <= 0.03 \
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
                    action, state_norm, desired_goal_norm = self.agent.choose_action(state, desired_goal,
                                                                                     train_mode=train_mode)
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

    @staticmethod
    def load_pickle(file_path):
        try:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                return data
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def run_episode_E2E_stack(self, train_mode, randomisation=False, withPause=False, real=False, reset=True,
                              display=False, success_threshold=0.003, num_steps=11, pred_horizon=1, obs_horizon=1,
                              action_horizon=1):
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
            "terminals": [],
            'trajectory': [],
            'error': [],
        }
        friction_change_times = 0
        per_success_rate = []
        episode_reward = 0
        print("Random: ", randomisation)
        terminated = None
        control_mode = None
        pos_idx = None
        next_env_dict = None
        info_ = None
        r = None
        t = None

        ''' Reset, pick up if needed '''
        inAir = False
        last_friction = 0
        slide_success_real = 0
        steps = 0
        env_dict, _ = self.reset(real=real, reset=reset, inAir=inAir, E2E=True)
        state, achieved_goal, desired_goal = self.extract_env_info(env_dict)

        ''' RANDOMISATION '''
        if randomisation:
            print("Randomisation: ", state[0], state[2])
            joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
            state[0] += joint_noise[0]
            state[2] += joint_noise[1]

        ''' Include action in obs '''
        if self.include_action_in_obs:
            state = np.concatenate([state, np.zeros(2)])

        ''' Stack obs '''
        if obs_horizon != 1:
            obs_deque = collections.deque([state] * obs_horizon, maxlen=obs_horizon)

        ''' Step '''
        # TODO: Uncomment below for desired task evaluation:
        #   1. Random policies with acceleration and deceleration, manually specified trajectories
        #   2. RL policies-collected demos with rotation (big step)
        ' -1- '
        # for t in range(500):
        ' -2- '
        while steps <= 20:
            ''' Choose action '''
            if obs_horizon != 1:
                # print(obs_deque)
                obs_seq = np.stack(obs_deque)
                nobs = torch.from_numpy(obs_seq).to('cpu', dtype=torch.float32)
                state = nobs.unsqueeze(0).flatten(start_dim=1)[0]
                # print(state)

            naction, _, _ = self.agent.choose_action(state,
                                                     desired_goal[:-2],
                                                     train_mode=train_mode)
            # naction = naction.detach().to('cpu').numpy()
            # naction = naction[0]
            # action_pred = unnormalize_data(naction, stats=stats['action'])
            action_pred = naction

            # only take action_horizon number of actions
            # TODO: Add this to force action horizon to be 1
            action_horizon = 1
            start = obs_horizon - 1
            end = start + action_horizon
            action_pred = np.array(action_pred).reshape(pred_horizon, 2)
            action = action_pred[start:end, :]

            for i in range(len(action)):
                friction_state, control_mode, _ = self.discretize_action_to_control_mode_E2E(action[i][1])
                if control_mode == 4 or control_mode == 5:
                    print(f"| Rotate with action {action[0], control_mode}")
                    num_steps = 30
                else:
                    num_steps = 11

                steps += 1
                if steps > 20:
                    break
                # input("Press to continue")
                # print(f"Action: {action, self.env.pos_idx}")

                ''' Slide '''
                if friction_state != last_friction:
                    friction_change_times += 1
                    next_env_dict, r, terminated, _, info_ = self.friction_change(friction_state, self.env)
                    last_friction = friction_state
                    start_joint_pos = next_env_dict['observation'][self.env.pos_idx * 2]
                    if terminated is False:
                        for num in range(num_steps):
                            next_env_dict, self.r_dict, terminated, _, info_ = self.env.step(action[i])
                            episode_dict['trajectory'].append(next_env_dict['observation'])
                            r = self.r_dict["E2E_IHM"]
                            if terminated is True:
                                # input("Press to continue")
                                print("Terminated during the step")
                            if self.r_dict["pos_control_position"] <= 0.03 \
                                    or self.r_dict["torque_control_position"] >= 1.65 \
                                    or terminated is True:
                                break

                            # if control_mode == 4 or control_mode == 5:
                            #     if abs((action[0]+1)/2*1.8807/7 - (start_joint_pos - next_env_dict['observation'][self.env.pos_idx*2])) < 0.01:
                            #         print("Rotate complete", num)
                            #         break
                            #     else:
                            #         print(num, (next_env_dict['observation'][self.env.pos_idx*2] - start_joint_pos), ((action[0]+1)/2)*1.8807/7)
                else:
                    start_joint_pos = state[self.env.pos_idx * 2]
                    for num in range(num_steps):
                        next_env_dict, self.r_dict, terminated, _, info_ = self.env.step(action[i])
                        episode_dict['trajectory'].append(next_env_dict['observation'])
                        r = self.r_dict["E2E_IHM"]
                        if terminated is True:
                            # input("Press to continue")
                            print("Terminated during the step")
                        if self.r_dict["pos_control_position"] <= 0.03 \
                                or self.r_dict["torque_control_position"] >= 1.65 \
                                or terminated is True:
                            break
                        # if control_mode == 4 or control_mode == 5:
                        #     if abs((action[0]+1)/2*1.8807/7 - (start_joint_pos - next_env_dict['observation'][self.env.pos_idx*2])) < 0.01:
                        #         print("Rotate complete", num)
                        #         break
                        #     else:
                        #         print(num, (next_env_dict['observation'][self.env.pos_idx*2] - start_joint_pos), ((action[0]+1)/2)*1.8807/7)

                # print(state)
                # episode_dict["state"].append(state[-1].copy())
                episode_dict["action"].append(action[i].copy())
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                episode_dict["desired_goal"].append(desired_goal.copy())

                next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(next_env_dict)
                if randomisation:
                    print("Randomisation: ", next_state[0], next_state[2])
                    joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
                    next_state[0] += joint_noise[0]
                    next_state[2] += joint_noise[1]

                if obs_horizon == 1:
                    state = next_state.copy()
                else:
                    obs_deque.append(next_state)
                achieved_goal = next_achieved_goal.copy()
                desired_goal = next_desired_goal.copy()

                if info_['is_success_old']:
                    print("| Slide Success")

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
                        # input("Press to continue")
                        break
                else:
                    ''' Real '''
                    pass

            if terminated is True or info_["is_success"] == 1:
                break

        if real:
            pass
        else:
            # episode_dict["state"].append(state.detach().clone())
            episode_dict["achieved_goal"].append(achieved_goal.copy())
            episode_dict["desired_goal"].append(desired_goal.copy())
            episode_dict["next_state"] = episode_dict["state"][1:]
            episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]

        slide_result = np.array(desired_goal - achieved_goal)
        print(f"------------------------------------{t}")

        return episode_dict, per_success_rate, episode_reward, slide_result, friction_change_times

    def run_episode_E2E(self,
                        train_mode,
                        randomisation=False,
                        real=False,
                        reset=True,
                        normalise=False,
                        object=None
                        ):
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
            "terminals": [],
            'trajectory': [],
            'error': [],
        }
        friction_change_times = 0
        per_success_rate = []
        episode_reward = 0
        print("Random: ", randomisation)
        terminated = None
        control_mode = None
        pos_idx = None
        next_env_dict = None
        info_ = None
        r = None
        t = None
        slide_succeed_in_this_episode = False

        # dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_50kdemos_slide'
        # dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos'
        # dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10k_demos_random_slide'
        # TODO: Use VFF-bigSteps-10000demos_slide to norm control mode
        # dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_50kdemos_slide_withJointDR'

        dataset_4_norm_path = None
        dataset_4_norm_path_rot = None
        dataset_4_norm_path_stop = None
        dataset_4_norm_path_mode = None
        if object == 'cube':
            ' Cube + no object velocity '
            # dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube/bigSteps_10000demos_slide_for_real'
            # dataset_4_norm_path_rot = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube_cylinder/train_bigSteps_10k_cube_cylinder_rotation_noObjVel'
            # dataset_4_norm_path_stop = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube/train_10k_cube_for_real_withRadi'
            pass

        if object == "cube_cylinder":
            ' Cube cylinder '
            # dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_cube_cylinder_slide'
            # dataset_4_norm_path_rot = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_cube_cylinder_rotation'
            # dataset_4_norm_path_stop = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_10k_cube_cylinder.pkl'

            ' Cube cylinder + No Object Velocity'
            # dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube_cylinder/train_bigSteps_10k_cube_cylinder_slide_noObjVel'
            # dataset_4_norm_path_rot = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube_cylinder/train_bigSteps_10k_cube_cylinder_rotation_noObjVel'
            # dataset_4_norm_path_stop = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube_cylinder/train_10k_cube_cylinder_noObjVel.pkl'
            # dataset_4_norm_path_mode = dataset_4_norm_path  # use this for other shapes

            ' Cube cylinder + No Object Velocity + No Joint Vel'
            dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube_cylinder/train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal'
            dataset_4_norm_path_rot = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube_cylinder/train_bigSteps_10k_cube_cylinder_rotation_noObjVel_forReal'
            dataset_4_norm_path_stop = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube_cylinder/train_10k_cube_cylinder_noObjVel_forReal'
            dataset_4_norm_path_mode = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube_cylinder/train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal'

        elif object == "three_cylinder":
            ' Three cylinder '
            dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_three_cylinder_slide_noObjVel'
            dataset_4_norm_path_rot = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_three_cylinder_rotation_noObjVel'
            dataset_4_norm_path_stop = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_10k_three_cylinder_noObjVel.pkl'

        elif object == "cube_1cm":
            ' Cube 1cm '
            dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube1cm/train_10k_cube_1cm_noObjVel_slide'
            dataset_4_norm_path_rot = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube1cm/train_10k_cube_1cm_noObjVel_rotation'
            dataset_4_norm_path_stop = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube1cm/train_10k_cube_1cm_noObjVel.pkl'

        elif object == "cube_2cm":
            ' Cube 2cm '
            dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube2cm/train_10k_cube_2cm_noObjVel_slide'
            dataset_4_norm_path_rot = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube2cm/train_10k_cube_2cm_noObjVel_rotation'
            dataset_4_norm_path_stop = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube2cm/train_10k_cube_2cm_noObjVel.pkl'

        elif object == "mixed_object":
            ' Mix object '
            dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/mixed_object/train_40k_noObjVel_slide'
            dataset_4_norm_path_rot = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/mixed_object/train_40k_noObjVel_rotation'
            dataset_4_norm_path_stop = '/Users/qiyangyan/Desktop/Diffusion/Demo_random/mixed_object/train_40k_noObjVel.pkl'

        if dataset_4_norm_path_mode is None:
            dataset_4_norm_path_mode = dataset_4_norm_path

        print("-" * 50)
        print("Object: ", object)
        print("dataset_4_norm_path: ", dataset_4_norm_path)
        print("dataset_4_norm_path_rot: ", dataset_4_norm_path_rot)
        print("dataset_4_norm_path_stop: ", dataset_4_norm_path_stop)
        print("dataset_4_norm_path_mode: ", dataset_4_norm_path_mode)
        print("-" * 50)

        ''' normalization '''
        # if normalise:
        with open(dataset_4_norm_path, 'rb') as f:
            dataset_4_norm = pickle.load(f)
        with open(dataset_4_norm_path_rot, 'rb') as f:
            dataset_4_norm_rot = pickle.load(f)
        with open(dataset_4_norm_path_stop, 'rb') as f:
            dataset_4_norm_stop = pickle.load(f)
        with open(dataset_4_norm_path_mode, 'rb') as f:
            dataset_4_norm_mode = pickle.load(f)

        goal_mean = np.mean(dataset_4_norm['desired_goals'], axis=0)
        goal_std = np.std(dataset_4_norm['desired_goals'], axis=0)
        obs_mean = np.mean(dataset_4_norm['observations'], axis=0)
        obs_std = np.std(dataset_4_norm['observations'], axis=0)

        goal_mean_rot = np.mean(dataset_4_norm_rot['desired_goals'], axis=0)
        goal_std_rot = np.std(dataset_4_norm_rot['desired_goals'], axis=0)
        obs_mean_rot = np.mean(dataset_4_norm_rot['observations'], axis=0)
        obs_std_rot = np.std(dataset_4_norm_rot['observations'], axis=0)

        goal_mean_stop = np.mean(dataset_4_norm_stop['desired_goals'], axis=0)
        goal_std_stop = np.std(dataset_4_norm_stop['desired_goals'], axis=0)
        obs_mean_stop = np.mean(dataset_4_norm_stop['observations'], axis=0)
        obs_std_stop = np.std(dataset_4_norm_stop['observations'], axis=0)

        goal_mean_mode = np.mean(dataset_4_norm_mode['desired_goals'], axis=0)
        goal_std_mode = np.std(dataset_4_norm_mode['desired_goals'], axis=0)
        obs_mean_mode = np.mean(dataset_4_norm_mode['observations'], axis=0)
        obs_std_mode = np.std(dataset_4_norm_mode['observations'], axis=0)

        ''' Reset, pick up if needed '''
        inAir = False
        last_friction = 0
        slide_success_real = 0
        env_dict, _ = self.reset(real=real, reset=reset, inAir=inAir, E2E=True)
        if real is True:
            _, _, _, state, achieved_goal = self.extract_env_info(env_dict, real=real)
            desired_goal = env_dict['demo_desired_goal']
        else:
            state, achieved_goal, desired_goal = self.extract_env_info(env_dict)

        ''' RANDOMISATION '''
        if randomisation:
            print("Randomisation: ", state[0], state[2])
            joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
            state[0] += joint_noise[0]
            state[2] += joint_noise[1]

        ''' Include action in obs '''
        if self.include_action_in_obs:
            state = np.concatenate([state, np.zeros(2)])

        ''' Step '''
        # TODO: Uncomment below for desired task evaluation:
        #   1. Random policies with acceleration and deceleration, manually specified trajectories
        #   2. RL policies-collected demos with rotation (big step)
        ' -1- '
        # for t in range(500):
        ' -2- '
        if normalise:
            print('-' * 50)
            print("| Normalize")
            print('-' * 50)
        print('-' * 50)
        if self.terminate_indicator is True:
            print("| Terminate Indicator: ", self.terminate_indicator)
        if self.control_mode_indicator is True:
            print("| Control Mode Indicator: ", self.terminate_indicator)
        print('-' * 50)
        for t in range(20):
            # print("Check state: ", state)

            # state_mode_stop = state.copy()
            # desired_goal_mode_stop = desired_goal.copy()
            state, desired_goal = remove_useless_data(state.copy(), desired_goal.copy(), real)
            # print(state, desired_goal)
            state_rot = state.copy()
            desired_goal_rot = desired_goal.copy()
            state_mode_stop = state.copy()
            desired_goal_mode_stop = desired_goal.copy()

            if normalise:
                ''' Normalize desired goal and action '''
                if len(desired_goal) == 11:
                    desired_goal_policy_input = normalize(desired_goal[:-2], goal_mean, goal_std)
                else:
                    assert len(desired_goal) == 9, f"check: {len(desired_goal)}"
                    desired_goal_policy_input = normalize(desired_goal, goal_mean, goal_std)
                state_policy_input = normalize(state, obs_mean, obs_std)
            else:
                # Big Step
                desired_goal_policy_input = desired_goal[:-2]
                state_policy_input = state

                # old
                # desired_goal_policy_input = desired_goal
                # state_policy_input = state

            # state, desired_goal = remove_useless_data(state.copy(), desired_goal.copy())
            # state_rot = state.copy()
            # desired_goal_rot = desired_goal.copy()


            ''' Choose action '''
            if len(goal_mean_stop) == 9:
                pass
            else:
                goal_mean_stop = goal_mean_stop[:-2]
                goal_std_stop = goal_std_stop[:-2]

            if self.terminate_indicator is True:
                if len(desired_goal_mode_stop) == 11:
                    # print(state.shape, obs_mean_stop.shape, desired_goal.shape)
                    slide_terminate = self.predict_terminate(normalize(state_mode_stop, obs_mean_stop, obs_std_stop),
                                                             normalize(desired_goal_mode_stop[:-2], goal_mean_stop,
                                                                       goal_std_stop))
                else:
                    assert len(desired_goal) == 9, f"check: {len(desired_goal)}"
                    slide_terminate = self.predict_terminate(normalize(state_mode_stop, obs_mean_stop, obs_std_stop),
                                                             normalize(desired_goal_mode_stop, goal_mean_stop, goal_std_stop))
                print("Slide complete: ", slide_terminate)

            if self.control_mode_indicator is True:
                if len(desired_goal_mode_stop) == 11:
                    action_control_mode = self.predict_control_mode(normalize(state_mode_stop, obs_mean_mode, obs_std_mode),
                                                                    normalize(desired_goal_mode_stop[:-2], goal_mean_mode, goal_std_mode))
                else:
                    assert len(desired_goal) == 9, f"check: {len(desired_goal)}"
                    action_control_mode = self.predict_control_mode(normalize(state_mode_stop, obs_mean_mode, obs_std_mode),
                                                                    normalize(desired_goal_mode_stop, goal_mean_mode, goal_std_mode))
                action_control_mode = action_control_mode / 6  # 0~3 -> 0~1
                action_control_mode = action_control_mode * 2 - 1  # 0 ~ 1 -> -1 ~ 1

            if t == 0:
                action, state_norm, desired_goal_norm = self.agent.choose_action(state_policy_input,
                                                                                 desired_goal_policy_input,
                                                                                 train_mode=train_mode)
                if self.control_mode_indicator is True:
                    if len(action) == 1:
                        action = np.append(action, action_control_mode)
                    else:
                        action[1] = action_control_mode  # convert from 0~3 to -1~1
                friction_state, control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])
            elif self.separate_policies is True and self.terminate_indicator is True and slide_terminate == 1:
                print("| Auto Terminate")
                state_policy_input = normalize(state_rot, obs_mean_rot, obs_std_rot)
                if len(desired_goal_rot) == 9:
                    desired_goal_policy_input = normalize(desired_goal_rot, goal_mean_rot, goal_std_rot)
                else:
                    desired_goal_policy_input = normalize(desired_goal_rot[:-2], goal_mean_rot, goal_std_rot)

                action, state_norm, desired_goal_norm = self.agent_rotation.choose_action(state_policy_input,
                                                                                          desired_goal_policy_input,
                                                                                          train_mode=train_mode)
                # print(action)
                action[0] = action[0] * 2 * 7 / 1.8807 - 1
                # action[1] = np.clip(action[1], 4 / 6, 1) * 2 - 1  # TODO: FOR the first cube policy
                friction_state, control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])
            elif self.separate_policies is True and self.terminate_indicator is False:
                if info_["is_success_old"] == 1:
                    print("Separate")
                    action, state_norm, desired_goal_norm = self.agent_rotation.choose_action(state_policy_input,
                                                                                              desired_goal_policy_input,
                                                                                              train_mode=train_mode)
                    # print(action)
                    action[0] = action[0] * 2 * 7 / 1.8807 - 1
                    action[1] = np.clip(action[1], 4 / 6, 1) * 2 - 1
                    friction_state, control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])
                else:
                    # This function contains merge state and action
                    action, friction_state, control_mode, state_norm, desired_goal_norm = \
                        self.choose_action_with_filter_E2E(state_policy_input,
                                                           desired_goal_policy_input,
                                                           self.r_dict,
                                                           t,
                                                           control_mode,
                                                           action,
                                                           train_mode=train_mode)
                    if self.control_mode_indicator is True:
                        if len(action) == 1:
                            action = np.append(action, action_control_mode)
                            friction_state, control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])
                        else:
                            action[1] = action_control_mode
                            friction_state, control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])
            else:
                # This function contains merge state and action
                action, friction_state, control_mode, state_norm, desired_goal_norm = \
                    self.choose_action_with_filter_E2E(state_policy_input,
                                                       desired_goal_policy_input,
                                                       self.r_dict,
                                                       t,
                                                       control_mode,
                                                       action,
                                                       train_mode=train_mode)
                if self.control_mode_indicator is True:
                    if len(action) == 1:
                        action = np.append(action, action_control_mode)
                        friction_state, control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])
                    else:
                        action[1] = action_control_mode
                        friction_state, control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])

            if control_mode == 4 or control_mode == 5:
                print(f"| Rotate with action {action[0], control_mode}")
                num_steps = 30
            else:
                num_steps = 11


            ''' Slide '''
            if real is True:
                display = False
                if friction_state != last_friction:
                    friction_change_times += 1
                    friction_action_2 = [2, friction_state, True]
                    self.real_env.change_friction_real(friction_action_2)
                    last_friction = friction_state
                for _ in range(11):
                    pos_idx = self.real_env.step_real(action)
                    next_env_dict = self.real_env.get_obs_real(display=display)
                    terminated = self.real_env.compute_terminate(next_env_dict['achieved_goal'])
                    if t < 2:
                        terminated = False
                    if next_env_dict["observation"][pos_idx * 2] <= 0.03 \
                            or next_env_dict["observation"][(1 - pos_idx) * 2] >= 1.65:
                        break
            else:
                if friction_state != last_friction:
                    friction_change_times += 1
                    next_env_dict, r, terminated, _, info_ = self.friction_change(friction_state, self.env)
                    last_friction = friction_state
                    start_joint_pos = next_env_dict['observation'][self.env.pos_idx * 2]
                    if terminated is False:
                        for num in range(num_steps):
                            next_env_dict, self.r_dict, terminated, _, info_ = self.env.step(action)
                            episode_dict['trajectory'].append(next_env_dict['observation'])
                            r = self.r_dict["E2E_IHM"]
                            if terminated is True:
                                # input("Press to continue")
                                print("Terminated during the step")
                            if self.r_dict["pos_control_position"] <= 0.03 \
                                    or self.r_dict["torque_control_position"] >= 1.65 \
                                    or terminated is True:
                                break
                else:
                    start_joint_pos = state[self.env.pos_idx * 2]
                    for num in range(num_steps):
                        next_env_dict, self.r_dict, terminated, _, info_ = self.env.step(action)
                        episode_dict['trajectory'].append(next_env_dict['observation'])
                        r = self.r_dict["E2E_IHM"]
                        if terminated is True:
                            # input("Press to continue")
                            print("Terminated during the step")
                        if self.r_dict["pos_control_position"] <= 0.03 \
                                or self.r_dict["torque_control_position"] >= 1.65 \
                                or terminated is True:
                            break

            episode_dict["state"].append(state.copy())
            episode_dict["action"].append(action.copy())
            episode_dict["achieved_goal"].append(achieved_goal.copy())
            episode_dict["desired_goal"].append(desired_goal.copy())

            print("Check")
            if real is True:
                _, _, _, next_state, next_achieved_goal = self.extract_env_info(next_env_dict, real=real)
                next_desired_goal = next_env_dict['demo_desired_goal']
            else:
                next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(next_env_dict)
            if randomisation:
                print("Randomisation: ", next_state[0], next_state[2])
                joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
                next_state[0] += joint_noise[0]
                next_state[2] += joint_noise[1]

            achieved_goal = next_achieved_goal.copy()
            desired_goal = next_desired_goal.copy()
            state = next_state.copy()

            if self.terminate_indicator is False:
                if info_['is_success_old'] == 1:
                    print("| Slide Success")
                    slide_succeed_in_this_episode = True

            if train_mode:
                ''' Train '''
                if terminated is True:
                    break
            elif not real:
                ''' Play '''
                per_success_rate.append(info_['is_success'])
                episode_reward += r
                if info_["is_success"] == 1:
                    print("| --------- Success: ", info_["is_success"], "--------- |")
                    # input("Press to continue")
                    break
                elif terminated is True:
                    print("Terminate: ", terminated)
                    break
                # elif info_["is_success_old"] == 1:
                #     print("Success: ", info_["is_success_old"])
                #     per_success_rate.append(info_['is_success_old'])
                #     # input("Press to continue")
                #     break
            else:
                ''' Real '''
                pass

        if real:
            pass
        else:
            episode_dict["state"].append(state.copy())
            episode_dict["achieved_goal"].append(achieved_goal.copy())
            episode_dict["desired_goal"].append(desired_goal.copy())
            episode_dict["next_state"] = episode_dict["state"][1:]
            episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]

        slide_result = np.array(desired_goal - achieved_goal)
        print(f"------------------------------------{t}")

        time.sleep(0.5)

        return episode_dict, per_success_rate, episode_reward, slide_result, friction_change_times, slide_succeed_in_this_episode

    def choose_action_with_filter_E2E(self, state, desired_goal, reward_dict, t, control_mode, action, train_mode):
        if self.include_action_in_obs:
            state = np.concatenate([state, action])
        action, state_norm, desired_goal_norm = self.agent.choose_action(state, desired_goal, train_mode=train_mode)
        new_control_mode = None
        friction_state = None
        if len(action) == 2:
            friction_state, new_control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])
            ''' Action feasibility filter '''
            # print(reward_dict)
            if reward_dict["pos_control_position"] <= 0.03 \
                    or reward_dict["pos_control_position"] >= 1.65 \
                    or reward_dict["torque_control_position"] >= 1.65 \
                    or reward_dict["torque_control_position"] <= 0.03 \
                    or ((new_control_mode == 4 or new_control_mode == 5) and action[0] < 0) \
                    or ((new_control_mode == 4 or new_control_mode == 5) and self.separate_policies is True) \
                    and t > 0:
                i = 0
                while True:
                    i += 1
                    if i > 10:
                        action[1] = np.random.uniform(-1, 1)
                    friction_state, new_control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])
                    # print("Check: ", new_control_mode, control_mode, action[0])
                    if control_mode == new_control_mode \
                            or (control_mode == 0 and new_control_mode == 3) \
                            or (control_mode == 1 and new_control_mode == 2) \
                            or (control_mode == 2 and new_control_mode == 1) \
                            or ((new_control_mode == 4 or new_control_mode == 5) and action[0] < 0) \
                            or (control_mode == 3 and new_control_mode == 0) \
                            or ((new_control_mode == 4 or new_control_mode == 5) and self.separate_policies is True):
                        # TODO: Uncomment "or ((new_control_mode == 4 or new_control_mode == 5) and action[0] < 0) \" for bigSteps
                        action, state_norm, desired_goal_norm = self.agent.choose_action(state, desired_goal,
                                                                                         train_mode=train_mode)
                    else:
                        return action, friction_state, new_control_mode, state_norm, desired_goal_norm
        return action, friction_state, new_control_mode, state_norm, desired_goal_norm

    def predict_terminate(self, state_norm, goal_norm):
        robot_qpos = state_norm[0:4]
        object_qpos = state_norm[8:15]
        inputs = np.concatenate((robot_qpos, object_qpos, goal_norm))
        # print(inputs.shape, robot_qpos.shape, object_qpos.shape, goal_norm.shape)
        inputs = torch.from_numpy(inputs).float().unsqueeze(0)

        self.agent_terminate.eval()
        with torch.no_grad():
            outputs_test = self.agent_terminate(inputs)
            predictions = torch.argmax(outputs_test, axis=1)
        return predictions

    def predict_control_mode(self, state_norm, goal_norm):
        robot_qpos = state_norm[0:4]
        object_qpos = state_norm[8:15]
        # inputs = np.concatenate((robot_qpos, object_qpos, goal_norm[:-2]))
        inputs = np.concatenate((robot_qpos, object_qpos, goal_norm))
        inputs = torch.from_numpy(inputs).float().unsqueeze(0)

        self.agent_control_mode.eval()
        with torch.no_grad():
            outputs_test = self.agent_control_mode(inputs)
            prediction = torch.argmax(outputs_test, axis=1)
        return prediction


if __name__ == "__main__":
    ''' RL with framework, workable version in sim and real '''
    env_name = "VariableFriction-v5"
    real_ = True
    display_ = False
    keep_reset_to_pick_up_pos = False
    render = True
    demontration_file_name_ = "VFF-3"
    # policy_path_ = '/Users/qiyangyan/Desktop/Training Files/Real4/Real4/pretrained_policy/VariableFriction.pth'  # 0.003
    # policy_path_ = '/Users/qiyangyan/Desktop/TrainingFiles/Trained Policy/Training4_2mm_DR/VariableFriction_3_24.pth'
    policy_path_ = '/Users/qiyangyan/Desktop/TrainingFiles/Trained Policy/Training3_5mm_DR/VariableFriction.pth'  # 0.005
    # policy_path_ = '/Users/qiyangyan/Downloads/diffusion_imitation-master/models/actor_1000.pth'

    # env_name = "VariableFriction-v6"
    output_dir_ = '/Users/qiyangyan/Desktop/Training Files/Real4/Real4/diffusion_policies/models_5'
    model_idx_ = 1200
    # with open(f'demonstration/VFF-1817demos', 'rb') as f:
    #     dataset = pickle.load(f)
    # for key in dataset.keys():
    #     print(np.shape(dataset[key]))
    trainer_evaluator = TrainEvaluateAgent(env_name,
                                           render=render,
                                           real=real_,
                                           display=display_,
                                           diffusion=False,  # choose diffusion agent
                                           output_dir=output_dir_,
                                           model_idx=model_idx_
                                           )
    trainer_evaluator.play(10,
                           randomise=False,
                           withRotation=True,
                           withPause=False,
                           real=real_,
                           display=display_,
                           keep_reset=keep_reset_to_pick_up_pos,
                           store_error=False,
                           collect_demonstration=True,
                           demontration_file_name=demontration_file_name_,
                           policy_path=policy_path_,
                           success_threshold=0.005,
                           )

    # trainer_evaluator.calculate_norm_parameter()

    '''End2End RL'''
    # env_name = "VariableFriction-v6"
    # real_ = False
    # display_ = False
    # output_dir = None
    # model_idx = None
    # dataset = None
    # trainer_evaluator = TrainEvaluateAgent(env_name,
    #                                        render=False,
    #                                        # render=False,
    #                                        real=real_,
    #                                        display=display_,
    #                                        local=True)
    # # trainer_evaluator.agent.load_weights_play()
    # trainer_evaluator.train(randomise=False, E2E=True)
    # # trainer_evaluator.eval_agent_E2E(randomise=False)
    # # print(trainer_evaluator.env.data.qpos)
    # # print(len(trainer_evaluator.env.data.qpos))
    # # print(len(trainer_evaluator.env.data.qvel))
    # #
    # # # while True:
    # # #     trainer_evaluator.reset_environment()
    # # #     input(f"Press, goal: {trainer_evaluator.real_env.convert_quat_to_euler(trainer_evaluator.env.goal[:7])}")

    '''Diffusion Evaluation'''
    # env_name = "VariableFriction-v5"
    # real_ = False
    # display_ = False
    # keep_reset_to_pick_up_pos = True
    # save_demontration_file_name_ = None
    # policy_path_ = 'None'
    # output_dir = '/Users/qiyangyan/Desktop/Training Files/Real4/Real4/diffusion_policies/models_2'
    # model_idx = 3000
    # with open(f'demonstration/VFF-1686demos', 'rb') as f:
    #     dataset = pickle.load(f)

    ''' Remember to modify the gym self.r_threshold '''
    real_ = False
    display_ = False
    keep_reset_to_pick_up_pos = True
    save_demontration_file_name_ = None
    policy_path_ = 'None'

    # TODO: Uncomment this for model_4
    # env_name = "VariableFriction-v7"
    # output_dir = '/Users/qiyangyan/Desktop/Training Files/Real4/Real4/diffusion_policies/models_4'
    # model_idx = 6000
    # with open(f'demonstration/VFF-1817demos', 'rb') as f:
    #     dataset = pickle.load(f)
    # for key in dataset.keys():
    #     print(np.shape(dataset[key]))

    # TODO: Uncomment this for big steps
    # env_name = "VariableFriction-v7"
    # output_dir = '/Users/qiyangyan/Desktop/Diffusion/Policies'
    # model_idx = 10000
    #
    # trainer_evaluator = TrainEvaluateAgent(env_name,
    #                                        render=True,
    #                                        real=real_,
    #                                        display=display_,
    #                                        diffusion=True,  # choose diffusion agent
    #                                        )
    # trainer_evaluator.play(100,
    #                        randomise=False,
    #                        withRotation=False,
    #                        withPause=False,
    #                        real=real_,
    #                        display=display_,
    #                        keep_reset=keep_reset_to_pick_up_pos,
    #                        store_error=False,
    #                        collect_demonstration=False,
    #                        demontration_file_name=save_demontration_file_name_,
    #                        policy_path=policy_path_,
    #                        success_threshold=0.005,
    #                        )
