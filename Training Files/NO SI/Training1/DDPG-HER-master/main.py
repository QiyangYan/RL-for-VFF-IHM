import gymnasium as gym
from agent import Agent
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from play import Play
# import mujoco_py
import random
from mpi4py import MPI
import psutil
import time
from copy import deepcopy as dc
import os
import torch
from Friction_Change.friction_change import FRICTION
import sys


ENV_NAME = "VariableFriction-v1"
INTRO = False
Train = False
Play_FLAG = True
MAX_EPOCHS = 50
MAX_CYCLES = 50
num_updates = 60
MAX_EPISODES = 2
memory_size = 7e+5 // 50
batch_size = 1024
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.98
tau = 0.05
k_future = 4

test_env = gym.make(ENV_NAME)
state_shape = test_env.observation_space.spaces["observation"].shape
n_actions = test_env.action_space.shape[0]
n_goals = test_env.observation_space.spaces["desired_goal"].shape[0]
action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['IN_MPI'] = '1'


def pick_up(inAir, env):
    pick_up_action = [0, 2, False]
    # print("start picking")
    'The position position-controlled finger reaches the middle'
    while True:  # for _ in range(105):
        # pick_up_action[0] += 0.01
        pick_up_action[0] = 1.05
        state, reward, _, _, _ = env.step(np.array(pick_up_action))
        # time.sleep(0.5)
        # while not reward["action_complete"]:
        #     state, reward, _, _, _ = env.step(np.array(pick_up_action))
            # let self.pick_up = true
        if abs(reward["pos_control_position"] - 1.05) < 0.003:
            break

    'Wait until the torque-controlled finger reaches the middle'
    for _ in range(50):
        _, reward, _, _, _ = env.step(np.array(pick_up_action))
    # print("pick up complete --------")

    'Wait until the finger raised to air'
    lift_action = [0, 3, False]
    if inAir is True:
        print("Lifting the block")
        while True:  # for _ in range(120):
            _, reward, _, _, _ = env.step(np.array(lift_action))
            if reward["action_complete"]:
                break
    return reward

def discretize_action_to_control_mode(action):
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


def friction_change(friction_state, env):
    friction_action_1 = [2, 0, True]
    friction_action_2 = [2, friction_state, True]
    # input("Press Enter to continue...")
    new_obs, rewards, terminated, _, infos = FRICTION.change_friction_full_obs(np.array(friction_action_1), env)
    if terminated:
        print("terminate at friction change to high")
        return new_obs, rewards["RL_IHM"], terminated, _, infos
    # input("press")
    new_obs, rewards, terminated, _, infos = FRICTION.change_friction_full_obs(np.array(friction_action_2), env)
    if terminated:
        print("terminate at friction change to low")
    return new_obs, rewards["RL_IHM"], terminated, _, infos


def _compute_distance(a, b):
    ''' get pos difference and rotation difference
    left motor pos: 0.037012 -0.1845 0.002
    right motor pos: -0.037488 -0.1845 0.002
    '''
    assert len(a) == 3 and len(a) == len(b), f"a and b has wrong length: {len(a), len(b)}"
    a[2] = b[2]
    delta_pos = np.array(a) - np.array(b)
    d_pos = np.linalg.norm(delta_pos)
    return d_pos


def eval_agent(env_, agent_):
    total_success_rate = []
    running_r = []
    for ep in range(10):
        print("episode: ", ep)
        per_success_rate = []
        env_dictionary = env_.reset()[0]
        s = env_dictionary["observation"]
        ag = env_dictionary["achieved_goal"]
        g = env_dictionary["desired_goal"]
        while np.mean(abs(ag - g)) <= 0.02:  # if the radi diff is too small, reset
            env_dictionary = env_.reset()[0]
            s = env_dictionary["observation"]
            ag = env_dictionary["achieved_goal"]
            g = env_dictionary["desired_goal"]
        ep_r = 0

        # print("start")

        inAir_ = False
        last_friction_ = 0
        r_dict = pick_up(inAir_, env_)

        for t in range(20):
            # print(t)
            with torch.no_grad():
                a = agent_.choose_action(s, g, train_mode=False)

            ''' Action feasibility filter '''
            if r_dict["pos_control_position"] <= 0.03 \
                    or r_dict["pos_control_position"] >= 1.65 \
                    or r_dict["torque_control_position"] >= 1.65 \
                    or r_dict["torque_control_position"] <= 0.03 \
                    and t > 0:
                while True:
                    friction_state_, new_control_mode_ = discretize_action_to_control_mode(a[1])
                    # print("invalid action, take another action: ", a, control_mode_, new_control_mode_)
                    if control_mode_ == new_control_mode_ \
                            or (control_mode_ == 0 and new_control_mode_ == 3) \
                            or (control_mode_ == 1 and new_control_mode_ == 2) \
                            or (control_mode_ == 2 and new_control_mode_ == 1) \
                            or (control_mode_ == 3 and new_control_mode_ == 0):
                        a = agent_.choose_action(s, g)
                    else:
                        break

            friction_state_, control_mode_ = discretize_action_to_control_mode(a[1])
            if friction_state_ != last_friction_:
                observation_new, r, terminated_, _, info_ = friction_change(friction_state_, env_)
                last_friction_ = friction_state_
                if terminated_ is False:
                    for _ in range(10):
                        observation_new, r_dict, terminated_, _, info_ = env_.step(a)
                        r = r_dict["RL_IHM"]
                        if r_dict["pos_control_position"] <= 0.03 \
                                or r_dict["pos_control_position"] >= 1.65 \
                                or r_dict["torque_control_position"] >= 1.65 \
                                or r_dict["torque_control_position"] <= 0.03 \
                                or terminated_ is True:
                            # print("terminate at: ", reward_dict["pos_control_position"],
                            #       reward_dict["torque_control_position"])
                            break
            else:
                for _ in range(10):
                    observation_new, r_dict, terminated_, _, info_ = env_.step(a)
                    r = r_dict["RL_IHM"]
                    if r_dict["pos_control_position"] <= 0.03 \
                            or r_dict["pos_control_position"] >= 1.65 \
                            or r_dict["torque_control_position"] >= 1.65 \
                            or r_dict["torque_control_position"] <= 0.03 \
                            or terminated_ is True:
                        # print("terminate at: ", reward_dict["pos_control_position"],
                        #       reward_dict["torque_control_position"])
                        break

            s = observation_new['observation']
            g = observation_new['desired_goal']
            per_success_rate.append(info_['is_success'])
            ep_r += r

            if terminated_ is True:
                break

        total_success_rate.append(per_success_rate[-1])
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate)
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r, ep_r


if INTRO:
    print(f"state_shape:{state_shape[0]}\n"
          f"number of actions:{n_actions}\n"
          f"action boundaries:{action_bounds}\n"
          f"max timesteps:{test_env._max_episode_steps}")
    for _ in range(3):
        done = False
        test_env.reset()
        while not done:
            action = test_env.action_space.sample()
            test_state, test_reward, test_terminate, test_truncated, test_info = test_env.step(action)
            test_done = test_terminate or test_truncated
            # substitute_goal = test_state["achieved_goal"].copy()
            # substitute_reward = test_env.compute_reward(
            #     test_state["achieved_goal"], substitute_goal, test_info)
            # print("r is {}, substitute_reward is {}".format(r, substitute_reward))
            test_env.render()
    exit(0)

env = gym.make(ENV_NAME, render_mode="human")
# env = gym.make(ENV_NAME)
env.reset(seed=MPI.COMM_WORLD.Get_rank())
# env.seed(MPI.COMM_WORLD.Get_rank())
random.seed(MPI.COMM_WORLD.Get_rank())
np.random.seed(MPI.COMM_WORLD.Get_rank())
torch.manual_seed(MPI.COMM_WORLD.Get_rank())
agent = Agent(n_states=state_shape,
              n_actions=n_actions,
              n_goals=n_goals,
              action_bounds=action_bounds,
              capacity=memory_size,
              action_size=n_actions,
              batch_size=batch_size,
              actor_lr=actor_lr,
              critic_lr=critic_lr,
              gamma=gamma,
              tau=tau,
              k_future=k_future,
              env=dc(env))


print("Damping: ", env.unwrapped.model.dof_damping[1])
print("Armature: ", env.unwrapped.model.dof_armature[1])
print("Frictionless: ", env.unwrapped.model.dof_frictionloss[1])
print("Gainprm: ", env.unwrapped.model.actuator_gainprm[0][0])
print("Biasprm: ", env.unwrapped.model.actuator_biasprm[0][0])
print("Force Range: ", env.unwrapped.model.actuator_forcerange[0])
print("Floor Friction: ", env.model.geom_friction[1][0])


if Train:
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
                episode_dict = {
                    "state": [],
                    "action": [],
                    "info": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": []}
                env_dict = env.reset()[0]
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]
                while np.mean(abs(achieved_goal - desired_goal)) <= 0.02:
                    env_dict = env.reset()[0]
                    state = env_dict["observation"]
                    achieved_goal = env_dict["achieved_goal"]
                    desired_goal = env_dict["desired_goal"]

                ''' Pick up if terminated '''
                inAir = False
                last_friction = 0
                reward_dict = pick_up(inAir, env)

                for t in range(20):

                    action = agent.choose_action(state, desired_goal)

                    ''' Action feasibility filter '''
                    if reward_dict["pos_control_position"] <= 0.03 \
                            or reward_dict["pos_control_position"] >= 1.65 \
                            or reward_dict["torque_control_position"] >= 1.65 \
                            or reward_dict["torque_control_position"] <= 0.03 \
                            and t > 0:
                        while True:
                            friction_state, new_control_mode = discretize_action_to_control_mode(action[1])
                            # print("invalid action, take another action: ", action, control_mode, new_control_mode)
                            if control_mode == new_control_mode \
                                or (control_mode == 0 and new_control_mode == 3) \
                                or (control_mode == 1 and new_control_mode == 2) \
                                or (control_mode == 2 and new_control_mode == 1) \
                                or (control_mode == 3 and new_control_mode == 0):
                                action = agent.choose_action(state, desired_goal)
                            else:
                                break

                    ''' Change Friction '''
                    friction_state, control_mode = discretize_action_to_control_mode(action[1])
                    # print("Discrete action that represents different control mode", actions)
                    # friction_state, control_mode = discretize_action_to_control_mode(action[1])
                    # input("press")
                    if friction_state != last_friction:
                        next_env_dict, rewards, terminated, _, infos = friction_change(friction_state, env)
                        last_friction = friction_state
                        if terminated is False:
                            # input("press, friction change complete")
                            for _ in range(10):
                                next_env_dict, reward_dict, terminated, _, info = env.step(action)
                                reward = reward_dict["RL_IHM"]
                                if reward_dict["pos_control_position"] <= 0.03 \
                                        or reward_dict["pos_control_position"] >= 1.65 \
                                        or reward_dict["torque_control_position"] >= 1.65 \
                                        or reward_dict["torque_control_position"] <= 0.03 \
                                        or terminated is True:
                                    # print("terminate at: ", reward_dict["pos_control_position"],
                                    #       reward_dict["torque_control_position"])
                                    break
                    else:
                        for _ in range(10):
                            next_env_dict, reward_dict, terminated, _, info = env.step(action)
                            reward = reward_dict["RL_IHM"]
                            if reward_dict["pos_control_position"] <= 0.03 \
                                    or reward_dict["pos_control_position"] >= 1.65 \
                                    or reward_dict["torque_control_position"] >= 1.65 \
                                    or reward_dict["torque_control_position"] <= 0.03 \
                                    or terminated is True:
                                # print("terminate at: ", reward_dict["pos_control_position"],
                                #       reward_dict["torque_control_position"])
                                break
                    done = terminated

                    next_state = next_env_dict["observation"]
                    next_achieved_goal = next_env_dict["achieved_goal"]
                    next_desired_goal = next_env_dict["desired_goal"]

                    episode_dict["state"].append(state.copy())
                    episode_dict["action"].append(action.copy())
                    episode_dict["achieved_goal"].append(achieved_goal.copy())
                    episode_dict["desired_goal"].append(desired_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()

                    if done:
                        break

                print("------------------------------------", t)

                episode_dict["state"].append(state.copy())
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                episode_dict["desired_goal"].append(desired_goal.copy())
                episode_dict["next_state"] = episode_dict["state"][1:]
                episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                mb.append(dc(episode_dict))

            agent.store(mb)
            for n_update in range(num_updates):
                actor_loss, critic_loss = agent.train()
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss

            epoch_actor_loss += cycle_actor_loss / num_updates
            epoch_critic_loss += cycle_critic_loss /num_updates
            agent.update_networks()

            if (cycle+1) % 10 == 0 and cycle != 0:
                success_rate, running_reward, episode_reward = eval_agent(env, agent)
                if MPI.COMM_WORLD.Get_rank() == 0:
                    ram = psutil.virtual_memory()
                    t_success_rate.append(success_rate)
                    print(f"Epoch:{epoch}| "
                          f"Running_reward:{running_reward[-1]:.3f}| "
                          f"EP_reward:{episode_reward:.3f}| "
                          f"Memory_length:{len(agent.memory)}| "
                          f"Duration:{time.time() - start_time:.3f}| "
                          f"Actor_Loss:{actor_loss:.3f}| "
                          f"Critic_Loss:{critic_loss:.3f}| "
                          f"Success rate:{success_rate:.3f}| "
                          f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM"
                          )
                    agent.save_weights()

                if MPI.COMM_WORLD.Get_rank() == 0:

                    with SummaryWriter("logs") as writer:
                        for i, success_rate in enumerate(t_success_rate):
                            writer.add_scalar("Success_rate", success_rate, i)

                    plt.style.use('ggplot')
                    plt.figure()
                    plt.plot(np.arange(0, epoch * 5 + (cycle+1)/10), t_success_rate)
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

elif Play_FLAG:
    player = Play(env, agent, max_episode=100)
    player.evaluate()
