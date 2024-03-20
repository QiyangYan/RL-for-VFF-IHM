"""Showing an error of nothing inside the info buffer, so the print result is wierd"""
from Training1.common.callback import SaveOnBestTrainingRewardCallback
from stable_baselines3 import PPO

class FRICTION_RL_Baseline():
    def __init__(self, env):
        self.mb = []
        self.num_updates = 10
        self.cycle_actor_loss = 0
        self.cycle_critic_loss = 0
        self.episode_count = 0
        self.cycle_count = 0
        self.MAX_EPISODES = 2
        self.MAX_CYCLES = 50

        self.memory_size = 7e+5 // 50
        self.batch_size = 256
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3
        # gamma = 0.2
        self.gamma = 0.98
        self.tau = 0.05
        self.k_future = 4
        self.epoch_actor_loss = 0
        self.epoch_critic_loss = 0
        self.count = 0

        log_dir = "/Users/qiyangyan/Desktop/LearnFrictionChange/trained_agent"
        self.callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        self.model = PPO(env, verbose=1)

    def change_friction(self, action, env, pos_idx):
        # friction_episode_dict = {
        #     "state": [],
        #     "action": [],
        #     "info": [],
        #     "achieved_goal": [],
        #     "desired_goal": [],
        #     "next_state": [],
        #     "next_achieved_goal": []}

        # next_env_dict, distance, terminated, truncated, info = env.step(action)
        # if terminated == True:
        #     return False, distance, next_env_dict["observation"][pos_idx * 2]  # false meaning not complete
        # next_env_dict = env.friction_obs_process(next_env_dict)

        # state = next_env_dict["observation"]
        # achieved_goal = next_env_dict["achieved_goal"]
        # desired_goal = next_env_dict["desired_goal"]

        # env.friction_change_startEnd_indicator(True)

        # print("friction state ", action[1])
        self.model.learn(total_timesteps=10000, action=action[1], callback=self.callback)
        self.count += 1
        if self.count > 100:
            self.model.save("trained_friction_change")
        # env.friction_change_startEnd_indicator(False)

        next_env_dict, distance, terminated, truncated, info = env.step_slide(action)
        if terminated == True:
            return False, distance, next_env_dict["observation"][pos_idx * 2]  # false meaning not complete
        next_env_dict = env.friction_obs_process(next_env_dict)

        # print("distance within friction change", distance)

        return True, distance, next_env_dict["observation"][pos_idx * 2]


    def change_friction_rotation(self, action, env):
        # friction_episode_dict = {
        #     "state": [],
        #     "action": [],
        #     "reward": [],
        #     "info": [],
        #     "achieved_goal": [],
        #     "desired_goal": [],
        #     "next_state": [],
        #     "next_achieved_goal": []}

        # state = next_env_dict["observation"]
        # achieved_goal = next_env_dict["achieved_goal"]
        # desired_goal = next_env_dict["desired_goal"]

        # next_env_dict, distance, terminated, truncated, info = env.step(action)
        # if terminated == True:
        #     return False, distance, next_env_dict["observation"]  # false meaning not complete
        # next_env_dict = env.friction_obs_process(next_env_dict)

        self.model.learn(total_timesteps=10000, action=action[1], callback=self.callback)
        self.count += 1
        if self.count > 1000:
            self.model.save("trained_friction_change")
        #      "learn_robust_friction_change"
        # env.friction_change_startEnd_indicator(True)
        #
        # while not env.action_complete():
        #     friction_action, _states = self.model.predict(state, deterministic=True)
        #     # friction_action = self.agent.choose_action(state, desired_goal)
        #     action[0] = friction_action[0]
        #
        #     next_env_dict, distance, terminated, truncated, info = env.step(action)
        #
        #     if terminated == True:
        #         return False, distance, next_env_dict["observation"]  # false meaning not complete
        #
        #     next_state = next_env_dict["observation"]
        #     next_achieved_goal = next_env_dict["achieved_goal"]
        #     next_desired_goal = next_env_dict["desired_goal"]
        #     # print("Compare achieved and desired:", next_achieved_goal, next_desired_goal)
        #
        #     friction_episode_dict["state"].append(state.copy())
        #     friction_episode_dict["action"].append(action.copy())
        #     friction_episode_dict["achieved_goal"].append(achieved_goal.copy())
        #     friction_episode_dict["desired_goal"].append(desired_goal.copy())
        #
        #     state = next_state.copy()
        #     achieved_goal = next_achieved_goal.copy()
        #     desired_goal = next_desired_goal.copy()
        #
        # # next_env_dict, distance, _, _, _ = env.step(action)
        #
        # friction_episode_dict["action"].append(action.copy())
        # friction_episode_dict["state"].append(state.copy())
        # friction_episode_dict["achieved_goal"].append(achieved_goal.copy())
        # friction_episode_dict["desired_goal"].append(desired_goal.copy())
        # friction_episode_dict["next_state"] = friction_episode_dict["state"][1:]
        # friction_episode_dict["next_achieved_goal"] = friction_episode_dict["achieved_goal"][1:]
        # self.mb.append(dc(friction_episode_dict))
        #
        # # update
        # if self.episode_count < self.MAX_EPISODES - 1:
        #     self.episode_count += 1
        # else:
        #     print(self.mb)
        #     self.agent.store(self.mb)
        #     # env.friction_change_startEnd_indicator(True)
        #     self.update()
        #     self.episode_count = 0
        #
        # env.friction_change_startEnd_indicator(False)

        next_env_dict, distance, terminated, truncated, info = env.step_slide(action)
        if terminated == True:
            return False, distance, next_env_dict["observation"]  # false meaning not complete
        next_env_dict = env.friction_obs_process(next_env_dict)

        return True, distance, next_env_dict["observation"]

    # def update(self):
    #     self.cycle_count += 1
    #     for n_update in range(self.num_updates):
    #         # print("train")
    #         actor_loss, critic_loss = self.agent.train()
    #         self.cycle_actor_loss += actor_loss
    #         self.cycle_critic_loss += critic_loss
    #
    #         if self.cycle_count == self.MAX_CYCLES:
    #             self.cycle_count = 0
    #             self.epoch_actor_loss += self.cycle_actor_loss / self.num_updates
    #             self.epoch_critic_loss += self.cycle_critic_loss / self.num_updates
    #
    #         self.agent.update_networks()
