import numpy as np
import time


class ROTATION:  # this class contains functions for rotation
    def __init__(self, env):
        self.env = env
        self.direction = 0
        self.rotation_start_pos = 0
        self.rotate_end_pos = np.array([0, 0])
        self.friction_action1 = np.array([2, 0])

    @staticmethod
    def modify_action(action, state, pos_idx):
        angle = state[pos_idx * 2]
        action[0] = angle
        return action

    def start_rotation(self, action, rotation_precision, episode_dict):
        ''' start of rotation might be hard'''
        action[0] = np.clip((action[0] - 0.05), 0, 1.65)
        next_env_dict, reward_dict, terminated, truncated, info = self.env.step(action)
        if terminated is True:
            print("Terminated")
            return terminated, reward_dict, next_env_dict, episode_dict

        self.rotation_start_pos = reward_dict["pos_control_position"]

        # last_reward = reward
        reward = reward_dict["current_goal_centre_distance"] * 20
        start_reward = reward
        last_reward = reward

        # print("Start: ", action[0], reward_dict["pos_control_position"])
        while True:
            ''' - 0.005 * (start_reward-reward) / start_reward) is added to slows down the movement 
            when get close to the target'''
            # action[0] = np.clip((action[0] + 0.01 - 0.005 * (start_reward-reward) / start_reward), 0, 1.65)
            action[0] = np.clip((action[0] - 0.01), 0, 1.65)
            next_env_dict, reward_dict, terminated, truncated, info = self.env.step(np.array(action))
            if terminated:
                print("Terminated")
                return terminated, reward_dict, next_env_dict, episode_dict

            episode_dict['trajectory'].append(next_env_dict['observation'])

            while action[0] - reward_dict["pos_control_position"] > 0.1:
                # print(action[0], reward_dict["pos_control_position"])
                next_env_dict, reward_dict, terminated, truncated, info = self.env.step(np.array(action))
                if terminated:
                    print("Terminated")
                    return terminated, reward_dict, next_env_dict, episode_dict
            last_reward = reward
            reward = reward_dict["current_goal_centre_distance"]
            # input("Press")
            if last_reward < reward or reward < rotation_precision:
                break

        action[0] = reward_dict["pos_control_position"]
        next_env_dict, reward_dict, terminated, truncated, info = self.env.step(np.array(action))
        if terminated is True:
            print("Terminated")
            return terminated, reward_dict, next_env_dict, episode_dict

        # action[0] = reward_dict["pos_control_position"]
        self.rotate_end_pos = action
        # print("Action after rotation: ", action)
        return terminated, reward_dict, next_env_dict, episode_dict

    # def reverse_rotate(self, pos_idx):
    #     """ moves back to the rotation start position to get ready for the next episode """
    #     action = self.rotate_end_pos
    #     action[0] = np.clip((action[0] + 0.01 * 1), 0, 1.68)
    #     next_env_dict, distance, terminated, _, _ = self.env.step(np.array(action))
    #     if terminated is True:
    #         # action = self.modify_action(action, next_env_dict["observation"], pos_idx)
    #         return action, False
    #     # print("start with action: ", action)
    #     start_time = time.time()
    #     sign_start = np.sign(next_env_dict["observation"][pos_idx*2] - self.rotation_start_pos)
    #     while abs(next_env_dict["observation"][pos_idx*2] - self.rotation_start_pos) > 0.01:
    #         sign = np.sign(next_env_dict["observation"][pos_idx*2] - self.rotation_start_pos)
    #         while abs(next_env_dict["observation"][pos_idx*2] - action[0]) > 0.05:
    #             # print(abs(next_env_dict["observation"][pos_idx*2] - action[0]), action[0])
    #             next_env_dict, distance, terminated, _, _ = self.env.step_slide(np.array(action))
    #             if abs(next_env_dict["observation"][pos_idx*2] - self.rotation_start_pos) < 0.01:
    #                 # action = self.modify_action(action, next_env_dict["observation"], pos_idx)
    #                 return action, True
    #             elif time.time() - start_time > 10:
    #                 print("\033[91m| Get stuck at reverse rotation \033[0m")
    #                 # action = self.modify_action(action, next_env_dict["observation"], pos_idx)
    #                 return action, False
    #         if sign != sign_start:
    #             action[0] = np.clip((action[0] - 0.01 * 1), 0, 1.68)
    #         else:
    #             action[0] = np.clip((action[0] + 0.01 * 1), 0, 1.68)
    #     # action = self.modify_action(action, next_env_dict["observation"], pos_idx)
    #     # print("Reverse rotation complete with action: ", action)
    #     return action, True

    def reverse_rotate(self):
        """ moves back to the rotation start position to get ready for the next episode """
        action = self.rotate_end_pos
        action[0] = np.clip((action[0] - 0.01), 0, 1.65)
        next_env_dict, reward_dict, terminated, _, _ = self.env.step(np.array(action))
        if terminated is True:
            return action, False

        # print("start with action: ", action)
        start_time = time.time()
        sign_start = np.sign(reward_dict["pos_control_position"] - self.rotation_start_pos)

        while abs(reward_dict['pos_control_position'] - self.rotation_start_pos) > 0.01:
            sign = np.sign(reward_dict['pos_control_position'] - self.rotation_start_pos)

            # while abs(reward_dict['pos_control_position'] - action[0]) > 0.05:
            next_env_dict, reward_dict, terminated, _, _ = self.env.step(np.array(action))
            if abs(reward_dict['pos_control_position'] - self.rotation_start_pos) < 0.01:
                return action, True
            elif time.time() - start_time > 10:
                print("\033[91m| Get stuck at reverse rotation \033[0m")
                return action, False

            if sign != sign_start:
                action[0] = np.clip((action[0] + 0.01 * 1), 0, 1.65)
            else:
                action[0] = np.clip((action[0] - 0.01 * 1), 0, 1.65)

        return action, True
