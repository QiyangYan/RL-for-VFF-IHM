import time

import numpy as np

class FRICTION:
    def __init__(self):
        pass

    @staticmethod
    def change_friction(action, env, pos_idx):

        ''' one while loop might have an early-finish when the position of finger is reached
        the friction change might not be completed
        '''

        while True:
            next_env_dict, distance, terminated, truncated, info = env.step(action)
            # print("two finger position: ", next_env_dict['observation'][0], next_env_dict['observation'][2])
            if terminated is True:
                return False, distance, next_env_dict["observation"][pos_idx*2]  # false meaning not complete
            if distance["action_complete"]:
                break
        return True, distance, next_env_dict["observation"][pos_idx*2]

    def friction_change_to_high(self, env):
        friction_action_1 = [2, 0, True]
        # input("Press Enter to continue...")
        new_obs, rewards, terminated, _, infos = self.change_friction_full_obs(np.array(friction_action_1), env)
        if terminated:
            print("terminate at friction change to high")
        return new_obs, rewards, terminated, _, infos

    def friction_change_to_low(self, friction_state, env):
        friction_action_2 = [2, friction_state, True]
        new_obs, rewards, terminated, _, infos = self.change_friction_full_obs(np.array(friction_action_2), env)
        if terminated:
            print("terminate at friction change to low")
        return new_obs, rewards, terminated, _, infos

    def change_friction_full_obs(self, action, env):
        '''
        one while loop might have an early-finish when the position of finger is reached
        the friction change might not be completed
        '''
        start_t = time.time()
        while True:
            next_env_dict, distance, terminated, truncated, info = env.step(action)
            # print("two finger position: ", next_env_dict['observation'][0], next_env_dict['observation'][2])
            if terminated is True:
                return next_env_dict, distance, terminated, truncated, info
                # return False, distance, next_env_dict["observation"][pos_idx*2]  # false meaning not complete
            if distance["action_complete"]:
                break
            if time.time() - start_t > 2.5:
                terminated = True
                return next_env_dict, distance, terminated, truncated, info
        return next_env_dict, distance, terminated, truncated, info

    @staticmethod
    def change_friction_rotation(action, env):

        while True:
            next_env_dict, distance, terminated, _, _ = env.step(action)
            if terminated is True:
                return False, distance, next_env_dict["observation"]  # false meaning not complete
            if env.action_complete():
                break

        return True, distance, next_env_dict["observation"]
