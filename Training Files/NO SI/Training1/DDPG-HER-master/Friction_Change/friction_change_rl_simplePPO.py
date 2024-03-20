from PPO.train import PPO_train
from PPO.test import PPO_test

'This is a simplified version PPO'
class FRICTION_RL_simp():
    def __init__(self, env, env_name, train, train_withTrainedAgent, test_withUntrainedAgent, trained_Path=None):
        self.count = 0
        self.train = train
        if not self.train:
            self.model = PPO_test(env, env_name, test_withUntrainedAgent)
        else:
            self.model = PPO_train(env, env_name, train_withTrainedAgent, trained_Path)

    def change_friction(self, action, env, pos_idx):
        env.friction_change_startEnd_indicator(True)

        if not self.train:
            _, _, _, _, _ = self.model.test(action[1])
        else:
            _, _, _, _, _ = self.model.train(action[1])

        next_env_dict, distance, terminated, truncated, info = env.step_slide(action)
        # if terminated is True:
        #     return False, distance, next_env_dict["observation"][pos_idx * 2]  # false meaning not complete

        next_env_dict = env.friction_obs_process(next_env_dict)  # same as modify action, but get the new state

        env.friction_change_startEnd_indicator(False)

        if terminated is True:
            return False, distance, next_env_dict["observation"][pos_idx * 2]  # false meaning not complete
        else:
            return True, distance, next_env_dict["observation"][pos_idx * 2]

    def change_friction_rotation(self, action, env):

        env.friction_change_startEnd_indicator(True)

        if not self.train:
            _, _, _, _, _ = self.model.test(action[1])
        else:
            _, _, _, _, _ = self.model.train(action[1])

        next_env_dict, distance, terminated, truncated, info = env.step_slide(action)

        next_env_dict = env.friction_obs_process(next_env_dict)

        env.friction_change_startEnd_indicator(False)

        if terminated is True:
            return False, distance, next_env_dict["observation"]  # false meaning not complete
        else:
            return True, distance, next_env_dict["observation"]