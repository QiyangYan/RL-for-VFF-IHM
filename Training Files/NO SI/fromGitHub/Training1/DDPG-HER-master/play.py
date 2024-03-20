import torch
from torch import device
import numpy as np
import cv2
from gymnasium import wrappers
from Friction_Change.friction_change import FRICTION
# from mujoco import GlfwContext
import time

# GlfwContext(offscreen=True)

# from mujoco_py.generated import const

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

def pick_up(inAir, env):
    pick_up_action = [0, 2, False]
    # print("start picking")
    'The position position-controlled finger reaches the middle'
    while True:  # for _ in range(105):
        pick_up_action[0] += 0.01
        state, reward, _, _, _ = env.step(np.array(pick_up_action))
        # time.sleep(0.5)
        while not reward["action_complete"]:
            state, reward, _, _, _ = env.step(np.array(pick_up_action))
            # let self.pick_up = true
        if abs(reward["pos_control_position"] - 1.05) < 0.02:
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

class Play:
    def __init__(self, env, agent, max_episode=4):
        self.env = env
        # self.env = wrappers.Monitor(env, "./videos", video_callable=lambda episode_id: True, force=True)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.load_weights()
        self.agent.set_to_eval_mode()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):

        for _ in range(self.max_episode):
            env_dict = self.env.reset()[0]
            state = env_dict["observation"]
            achieved_goal = env_dict["achieved_goal"]
            desired_goal = env_dict["desired_goal"]
            while np.mean(abs(achieved_goal - desired_goal)) <= 0.02:
                env_dict = self.env.reset()[0]
                state = env_dict["observation"]
                achieved_goal = env_dict["achieved_goal"]
                desired_goal = env_dict["desired_goal"]

            # done = False
            episode_reward = 0

            inAir_ = False
            last_friction_ = 0
            r_dict = pick_up(inAir_, self.env)

            # while not done:
            for t in range(20):
                action = self.agent.choose_action(state, desired_goal, train_mode=False)

                ''' Action feasibility filter '''
                if r_dict["pos_control_position"] <= 0.03 \
                        or r_dict["pos_control_position"] >= 1.65 \
                        or r_dict["torque_control_position"] >= 1.65 \
                        or r_dict["torque_control_position"] <= 0.03 \
                        and t > 0:
                    while True:
                        friction_state_, new_control_mode_ = discretize_action_to_control_mode(action[1])
                        # print("invalid action, take another action: ", action, control_mode, new_control_mode)
                        if control_mode == new_control_mode_ \
                                or (control_mode == 0 and new_control_mode_ == 3) \
                                or (control_mode == 1 and new_control_mode_ == 2) \
                                or (control_mode == 2 and new_control_mode_ == 1) \
                                or (control_mode == 3 and new_control_mode_ == 0):
                            action = self.agent.choose_action(state, desired_goal)
                        else:
                            break

                friction_state_, control_mode = discretize_action_to_control_mode(action[1])
                if friction_state_ != last_friction_:
                    next_env_dict, r, terminated_, _, info_ = friction_change(friction_state_, self.env)
                    last_friction_ = friction_state_
                    if terminated_ is False:
                        for _ in range(10):
                            next_env_dict, r_dict, terminated_, _, info_ = self.env.step(action)
                            r = r_dict["RL_IHM"]
                            if r_dict["pos_control_position"] <= 0 \
                                    or r_dict["pos_control_position"] >= 1.68 \
                                    or r_dict["torque_control_position"] >= 1.65 \
                                    or r_dict["torque_control_position"] <= 0.03 \
                                    or terminated_ is True:
                                # print("terminate at: ", reward_dict["pos_control_position"],
                                #       reward_dict["torque_control_position"])
                                break
                else:
                    for _ in range(10):
                        next_env_dict, r_dict, terminated_, _, info_ = self.env.step(action)
                        r = r_dict["RL_IHM"]
                        if r_dict["pos_control_position"] <= 0 \
                                or r_dict["pos_control_position"] >= 1.68 \
                                or r_dict["torque_control_position"] >= 1.65 \
                                or r_dict["torque_control_position"] <= 0.03 \
                                or terminated_ is True:
                            # print("terminate at: ", reward_dict["pos_control_position"],
                            #       reward_dict["torque_control_position"])
                            break

                # next_env_dict, r, done, _ = self.env.step(action)

                print(t)
                if info_["is_success"]:
                    print("success")
                    # input("press")


                next_state = next_env_dict["observation"]
                next_desired_goal = next_env_dict["desired_goal"]
                episode_reward += r
                state = next_state.copy()
                desired_goal = next_desired_goal.copy()
                # I = self.env.render(mode="human")  # mode = "rgb_array
                # self.env.viewer.cam.type = const.CAMERA_FREE
                # self.env.viewer.cam.fixedcamid = 0
                # I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                # cv2.imshow("I", I)
                # cv2.waitKey(2)

                if terminated_ is True:
                    break

            print(f"episode_reward:{episode_reward:3.3f}")

        self.env.close()
