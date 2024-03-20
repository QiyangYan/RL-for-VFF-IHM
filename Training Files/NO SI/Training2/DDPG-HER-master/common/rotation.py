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

    def start_rotation(self, action):
        """ start friction change first """
        # print("friction changing")
        """ if target is on the left-hand-side of the object, rotate to left, Right is pos """
        action[0], pos_idx = self.env.switch_ctrl_type(0)

        # complete, reward, observation = friction_class.change_friction_rotation(np.array(self.friction_action1),
        #                                                                         self.env)
        # print("reward after both to high friction change", reward["d_radi"])
        # posAfterFrictionChange = observation[pos_idx*2]

        # action[0] = posAfterFrictionChange
        # if complete is False:
        #     print("\033[92m| Terminated during friction changing\033[0m")
        #     # time.sleep(5)
        #     return False, reward, pos_idx

        ''' start of rotation might be hard'''
        # print(reward)
        action[0] = np.clip((action[0] + 0.1), 0, 1.68)
        action[1] = 0
        # action[0] = action[0] + direction * 0.1
        next_env_dict, distance, terminated, truncated, info = self.env.step_slide(np.array(action))
        if terminated is True:
            reward = distance["d_pos"] * 20
            return False, reward, pos_idx
        print("reward after both to high friction change", distance["d_radi"])
        if pos_idx == 1:
            reverse_rotate_pos_idx = 0
        else:
            reverse_rotate_pos_idx = 1
        self.rotation_start_pos = next_env_dict["observation"][reverse_rotate_pos_idx*2]

        # last_reward = reward
        reward = distance["d_pos"] * 20
        start_reward = reward
        last_reward = reward
        # print(reward, last_reward)
        # rotation_complete = False
        # count = 0
        while reward <= last_reward:
            # print("Rotating")
            ''' - 0.005 * (start_reward-reward) / start_reward) is added to slows down the movement 
            when get close to the target'''
            action[0] = np.clip((action[0] + 0.01 - 0.005 * (start_reward-reward) / start_reward), 0, 1.68)
            next_env_dict, distance, terminated, truncated, info = self.env.step_slide(np.array(action))
            if terminated is True:
                reward = distance["d_pos"] * 20
                return False, reward, pos_idx
            last_reward = reward
            reward = distance["d_pos"] * 20
            # print(reward,last_reward,action[0])
        action = self.modify_action(action, next_env_dict["observation"], pos_idx)
        # print("end position of rotation: ", next_env_dict["observation"][0],next_env_dict["observation"][2])
        action[0], pos_idx = self.env.switch_ctrl_type_direct()
        # print("rotation end with pos_idx: ", pos_idx, action[0])
        next_env_dict, distance, terminated, truncated, info = self.env.step_slide(np.array(action))
        if terminated is True:
            reward = distance["d_pos"] * 20
            return False, reward, pos_idx
        # time.sleep(2)
        self.rotate_end_pos = action
        # print(self.rotate_end_pos)
        # time.sleep(5)
        return True, reward, pos_idx

    def reverse_rotate(self, pos_idx):
        """ moves back to the rotation start position to get ready for the next episode """
        action = self.rotate_end_pos
        # start_friction = action[1]
        # print("reverse rotate start pos: ", action)
        # print("reverse rotate goal pos: ", self.rotation_start_pos)
        action[0] = np.clip((action[0] + 0.01 * 1), 0, 1.68)
        action[1] = 0
        next_env_dict, distance, terminated, _, _ = self.env.step_slide(np.array(action))
        if terminated is True:
            action = self.modify_action(action, next_env_dict["observation"], pos_idx)
            return action, False
        # print("start with action: ", action)
        start_time = time.time()
        sign_start = np.sign(next_env_dict["observation"][pos_idx*2] - self.rotation_start_pos)
        while abs(next_env_dict["observation"][pos_idx*2] - self.rotation_start_pos) > 0.01:
            sign = np.sign(next_env_dict["observation"][pos_idx*2] - self.rotation_start_pos)
            while abs(next_env_dict["observation"][pos_idx*2] - action[0]) > 0.05:
                # print(abs(next_env_dict["observation"][pos_idx*2] - action[0]), action[0])
                next_env_dict, distance, terminated, _, _ = self.env.step_slide(np.array(action))
                if abs(next_env_dict["observation"][pos_idx*2] - self.rotation_start_pos) < 0.01:
                    action = self.modify_action(action, next_env_dict["observation"], pos_idx)
                    return action, True
                elif time.time() - start_time > 10:
                    print("\033[91m| Get stuck at reverse rotation \033[0m")
                    action = self.modify_action(action, next_env_dict["observation"], pos_idx)
                    return action, False
            if sign != sign_start:
                action[0] = np.clip((action[0] - 0.01 * 1), 0, 1.68)
            else:
                action[0] = np.clip((action[0] + 0.01 * 1), 0, 1.68)
        action = self.modify_action(action, next_env_dict["observation"], pos_idx)
        # print("Reverse rotation complete with action: ", action)
        return action, True
