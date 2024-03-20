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
from gymnasium_robotics.envs.training3.DomainRandomisation.randomisation import RandomisationModule

from agent import Agent
# from play import Play
# Assume friction_change.friction_change is a module you have that contains the change_friction logic
from Friction_Change.friction_change import FRICTION
from common.common import COMMON

to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024
MAX_EPOCHS = 50
MAX_CYCLES = 50
num_updates = 60
MAX_EPISODES = 2


class ActionUtility:
    def __init__(self, env):
        self.friction = FRICTION()
        self.action_functions = COMMON(env)

    @staticmethod
    def discretize_action_to_control_mode(action):
        # Your action discretization logic here
        action_norm = (action + 1) / 2
        if 1 / 4 > action_norm >= 0:
            control_mode = 0
            friction_state = 1
        elif 2 / 4 > action_norm > 1 / 4:
            control_mode = 1
            friction_state = 1
        elif 3 / 4 > action_norm > 2 / 4:
            control_mode = 2
            friction_state = -1
        else:
            assert 4 >= action_norm > 3 / 4
            control_mode = 3
            friction_state = -1
        return friction_state, control_mode
        pass

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


class EnvironmentSetup(ActionUtility):
    def __init__(self, env: str):
        # self.env_name = env_name_
        # self.env = self.initialize_env()
        self.env = env
        super().__init__(self.env)
        self.agent = self.initialize_agent()

    def initialize_env(self):
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["IN_MPI"] = "1"
        # env = gym.make(self.env_name, render_mode="human")
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
            capacity=7e5 // 50,
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
    def __init__(self, env_name_):
        super().__init__(env_name_)
        self.domain_randomise = RandomisationModule()

    def reset_environment(self):
        while True:
            env_dict = self.env.reset()[0]
            if np.mean(abs(env_dict["achieved_goal"] - env_dict["desired_goal"])) > 0.02:
                return env_dict

    def choose_action_with_filter(self, state, desired_goal, reward_dict, t, control_mode, train_mode):
        action = self.agent.choose_action(state, desired_goal, train_mode=train_mode)
        ''' Action feasibility filter '''
        # print(reward_dict)
        if reward_dict["pos_control_position"] <= 0.03 \
                or reward_dict["pos_control_position"] >= 1.65 \
                or reward_dict["torque_control_position"] >= 1.65 \
                or reward_dict["torque_control_position"] <= 0.03 \
                and t > 0:
            while True:
                friction_state, new_control_mode = self.discretize_action_to_control_mode(action[1])
                if control_mode == new_control_mode \
                        or (control_mode == 0 and new_control_mode == 3) \
                        or (control_mode == 1 and new_control_mode == 2) \
                        or (control_mode == 2 and new_control_mode == 1) \
                        or (control_mode == 3 and new_control_mode == 0):
                    action = self.agent.choose_action(state, desired_goal, train_mode=train_mode)
                else:
                    return action, friction_state, new_control_mode
        friction_state, new_control_mode = self.discretize_action_to_control_mode(action[1])
        return action, friction_state, new_control_mode

    @staticmethod
    def extract_env_info(env_dict):
        return env_dict["observation"], env_dict["achieved_goal"], env_dict["desired_goal"]

    def run_episode_with_reset(self, train_mode, randomisation=False):
        episode_dict = {
            "state": [],
            "action": [],
            "info": [],
            "achieved_goal": [],
            "desired_goal": [],
            "next_state": [],
            "next_achieved_goal": []}
        per_success_rate = []
        episode_reward = 0

        env_dict = self.reset_environment()
        state, achieved_goal, desired_goal = self.extract_env_info(env_dict)
        if randomisation:
            # print("Randomisation: ", state[0], state[2])
            joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
            state[0] += joint_noise[0]
            state[2] += joint_noise[1]
            # print("Randomisation: ", state[0], state[2])

        ''' Pick up if terminated '''
        inAir = False
        last_friction = 0
        reward_dict = self.action_functions.pick_up(inAir)
        # state, achieved_goal, desired_goal = self.extract_env_info(env_dict)

        for t in range(20):
            if t == 0:
                action = self.agent.choose_action(state, desired_goal, train_mode=train_mode)
                friction_state, control_mode = self.discretize_action_to_control_mode(action[1])
            else:
                action, friction_state, control_mode = self.choose_action_with_filter(state,
                                                                                      desired_goal,
                                                                                      reward_dict,
                                                                                      t,
                                                                                      control_mode,
                                                                                      train_mode=train_mode)
            ''' Change Friction '''
            if friction_state != last_friction:
                next_env_dict, r, terminated, _, info_ = self.friction_change(friction_state, self.env)
                last_friction = friction_state
                if terminated is False:
                    for _ in range(11):
                        next_env_dict, r_dict, terminated, _, info_ = self.env.step(action)
                        r = r_dict["RL_IHM"]
                        if r_dict["pos_control_position"] <= 0.03 \
                                or r_dict["torque_control_position"] >= 1.65 \
                                or terminated is True:
                            break
            else:
                for _ in range(11):
                    next_env_dict, r_dict, terminated, _, info_ = self.env.step(action)
                    r = r_dict["RL_IHM"]
                    if r_dict["pos_control_position"] <= 0.03 \
                            or r_dict["torque_control_position"] >= 1.65 \
                            or terminated is True:
                        break

            episode_dict["state"].append(state.copy())
            episode_dict["action"].append(action.copy())
            episode_dict["achieved_goal"].append(achieved_goal.copy())
            episode_dict["desired_goal"].append(desired_goal.copy())

            next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(next_env_dict)
            if randomisation:
                # print("Randomisation: ", next_state[0], next_state[2])
                joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
                next_state[0] += joint_noise[0]
                next_state[2] += joint_noise[1]
                # print("After Randomisation: ", next_state[0], next_state[2])

            state = next_state.copy()
            achieved_goal = next_achieved_goal.copy()
            desired_goal = next_desired_goal.copy()
            per_success_rate.append(info_['is_success'])
            episode_reward += r

            # print(f"Action at time {t} with pos index {self.env.pos_idx}: ",
            #       self.env.gripper_pos_ctrl,
            #       self.env.data.qpos[self.env.pos_idx * 2 + 1],
            #       self.env.data.qpos[self.env.pos_idx * 2 + 1] - self.env.start,
            #       (action[0] + 1) / 2 * 1.8807 / 5)

            # print(info_["is_success"])
            if train_mode:
                if terminated is True:
                    break
            else:
                if terminated is True:
                    print("Terminate: ", terminated)
                    break
                elif info_["is_success"] == 1:
                    # print("Success: ", info_["is_success"])
                    break

        print(f"------------------------------------{t}")

        episode_dict["state"].append(state.copy())
        episode_dict["achieved_goal"].append(achieved_goal.copy())
        episode_dict["desired_goal"].append(desired_goal.copy())
        episode_dict["next_state"] = episode_dict["state"][1:]
        episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]

        return episode_dict, per_success_rate, episode_reward

    def train(self, randomise):
        # Training logic here
        t_success_rate = []
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
                    episode_dict, _, _ = self.run_episode_with_reset(train_mode=True, randomisation=randomise)
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
            episode_dict, success, episode_reward = self.run_episode_with_reset(train_mode=False, randomisation=randomise)
            total_success_rate.append(success[-1])
            if ep == 0:
                running_r.append(episode_reward)
            else:
                running_r.append(running_r[-1] * 0.99 + 0.01 * episode_reward)

        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate)
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r, episode_reward

    def play(self, num_episodes, randomise):
        # play = Play(self.env, self.agent)
        self.agent.load_weights()
        self.agent.set_to_eval_mode()
        # self.device = device("cuda" if torch.cuda.is_available() else "cpu")
        num_success = 0
        for ep in range(num_episodes):
            # print("episode: ", ep)
            episode_dict, success, episode_reward = self.run_episode_with_reset(train_mode=False, randomisation=randomise)
            # print(f"episode_reward:{episode_reward:3.3f}")
            if success[-1]:
                num_success += 1
                print(f"---- Episode {ep} SUCCESS ---- ")
                # time.sleep(2)
            else:
                print(f"---- Episode {ep} Failed ---- ")
                # time.sleep(2)

            if (ep+1) % 10 == 0:
                print("Success rate: ", num_success/(ep+1))

        self.env.close()


if __name__ == "__main__":
    env_name = "VariableFriction-v3"

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["IN_MPI"] = "1"
    # env = gym.make(env_name, render_mode="human")
    env = gym.make(env_name)
    env.reset(seed=MPI.COMM_WORLD.Get_rank())
    random.seed(MPI.COMM_WORLD.Get_rank())
    np.random.seed(MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(MPI.COMM_WORLD.Get_rank())

    trainer_evaluator = TrainEvaluateAgent(env)

    # Example usage
    trainer_evaluator.train(randomise=True)
    # trainer_evaluator.eval_agent()
    # trainer_evaluator.play(100, randomise=False)
