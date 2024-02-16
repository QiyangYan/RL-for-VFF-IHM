
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

    @staticmethod
    def change_friction_full_obs(action, env):

        ''' one while loop might have an early-finish when the position of finger is reached
        the friction change might not be completed
        '''

        while True:
            next_env_dict, distance, terminated, truncated, info = env.step(action)
            # print("two finger position: ", next_env_dict['observation'][0], next_env_dict['observation'][2])
            if terminated is True:
                return next_env_dict, distance, terminated, truncated, info
            if distance["action_complete"]:
                break
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
