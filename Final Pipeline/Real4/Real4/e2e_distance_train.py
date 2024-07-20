from main import TrainEvaluateAgent
import numpy as np
import pickle
import matplotlib.pyplot as plt


def normalize(data, mean, std):
    # Ensure standard deviation is not zero to avoid division by zero error
    std = np.where(std == 0, 1, std)
    # print("normal", mean, std)
    return (data - mean) / std


class TrainE2E(TrainEvaluateAgent):
    def __init__(self, env_name, render, seed, include_action_in_obs, local, norm, randomise):
        super().__init__(
            env_name_=env_name,
            real=False,
            local=local,
            render=render,
            display=False,
            seed=seed,
            include_action_in_obs=include_action_in_obs,
        )
        # dataset_4_norm_path = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos'
        # with open(dataset_4_norm_path, 'rb') as f:
        #     dataset_4_norm = pickle.load(f)
        # self.goal_mean = np.mean(dataset_4_norm['desired_goals'], axis=0)[:-2]
        # self.goal_std = np.std(dataset_4_norm['desired_goals'], axis=0)[:-2]
        # self.obs_mean = np.mean(dataset_4_norm['observations'], axis=0)
        # self.obs_std = np.std(dataset_4_norm['observations'], axis=0)

        self.goal_mean = None
        self.goal_std = None
        self.obs_mean = None
        self.obs_std = None

        self.normalise = norm
        self.randomise = randomise

        print('-' * 50)
        print("| seed: ", self.seed_idx)
        print("| observation space: ", self.env.observation_space.spaces["observation"].shape)
        print("| action space: ", self.env.action_space.shape)
        print("| desired goal space: ", self.env.observation_space.spaces["desired_goal"].shape[0])
        print("| action bound: ", (self.env.action_space.low[0], self.env.action_space.high[0]))
        print("| normalize: ", self.normalise)
        print("| randomisation: ", self.randomise)
        print('-' * 50)

    def run_episode_E2E_test_distance_metric(self,
                                             train_mode,
                                             real=False,
                                             reset=True
                                             ):
        normalise = self.normalise
        randomisation = self.randomise
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
        # print("Random: ", randomisation)
        terminated = None
        control_mode = None
        pos_idx = None
        next_env_dict = None
        info_ = None
        r = None
        t = None
        slide_succeed_in_this_episode = False

        ''' Reset, pick up if needed '''
        inAir = False
        last_friction = 0
        slide_success_real = 0
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

        ''' Step '''
        # print("Start step")
        for t in range(20):
            if normalise:
                ''' Normalize desired goal and action '''
                desired_goal_policy_input = normalize(desired_goal, self.goal_mean, self.goal_std)
                state_policy_input = normalize(state, self.obs_mean, self.obs_std)
            else:
                # Big Step
                desired_goal_policy_input = desired_goal
                state_policy_input = state

            # assert len(desired_goal_policy_input) == 7, f"Check: {len(desired_goal_policy_input)}"
            # assert len(state_policy_input) == 24, f"Check: {len(state_policy_input)}"

            ''' Choose action '''
            if t == 0:
                # TODO: Modify this to include rotation
                action, state_norm, desired_goal_norm = self.agent.choose_action(state_policy_input,
                                                                                 desired_goal_policy_input,
                                                                                 train_mode=train_mode)
                friction_state, control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])
            elif real:
                pass
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

            ''' Slide '''
            if friction_state != last_friction:
                friction_change_times += 1
                next_env_dict, r, terminated, _, info_ = self.friction_change(friction_state, self.env)
                last_friction = friction_state
                if terminated is False:
                    for num in range(11):
                        next_env_dict, self.r_dict, terminated, _, info_ = self.env.step(action)
                        episode_dict['trajectory'].append(next_env_dict['observation'])
                        r = self.r_dict["E2E_IHM"]
                        if terminated is True:
                            print("Terminated during the step")
                        if self.r_dict["pos_control_position"] <= 0.03 \
                                or self.r_dict["torque_control_position"] >= 1.65 \
                                or terminated is True:
                            break
            else:
                for num in range(11):
                    next_env_dict, self.r_dict, terminated, _, info_ = self.env.step(action)
                    episode_dict['trajectory'].append(next_env_dict['observation'])
                    r = self.r_dict["E2E_IHM"]
                    if terminated is True:
                        print("Terminated during the step")
                    if self.r_dict["pos_control_position"] <= 0.03 \
                            or self.r_dict["torque_control_position"] >= 1.65 \
                            or terminated is True:
                        break

            episode_dict["state"].append(state.copy())
            episode_dict["action"].append(action.copy())
            episode_dict["achieved_goal"].append(achieved_goal.copy())
            episode_dict["desired_goal"].append(desired_goal.copy())
            episode_dict["reward"].append(r)

            next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(next_env_dict)
            if randomisation:
                print("Randomisation: ", next_state[0], next_state[2])
                joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
                next_state[0] += joint_noise[0]
                next_state[2] += joint_noise[1]

            achieved_goal = next_achieved_goal.copy()
            desired_goal = next_desired_goal.copy()
            state = next_state.copy()
            episode_reward += r
            per_success_rate.append(info_['is_success'])

            # print(r)

            if train_mode:
                ''' Train '''
                if terminated is True:
                    break
            elif not real:
                ''' Play '''
                if info_["is_success"] == 1:
                    print("| --------- Success: ", info_["is_success"], "--------- |")
                    # input("Press to continue")
                    break
                elif terminated is True:
                    print("Terminate: ", terminated)
                    break
            else:
                ''' Real '''
                pass
            # plt.figure(figsize=(10, 5))
            # plt.plot(episode_dict["reward"], color='r')
            # plt.title('Expert Demonstration Reward History')
            # plt.xlabel('Index')
            # plt.ylabel('Value')
            # plt.show()
            #
            # input("Press")

        if real:
            pass
        else:
            episode_dict["state"].append(state.copy())
            episode_dict["achieved_goal"].append(achieved_goal.copy())
            episode_dict["desired_goal"].append(desired_goal.copy())
            episode_dict["next_state"] = episode_dict["state"][1:]
            episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
            episode_dict["reward"].append(r)

        print(f"------------------------------------{t}")

        return episode_dict, per_success_rate, episode_reward, friction_change_times


if __name__ == '__main__':
    env_name_ = "VariableFriction-v6"
    render_ = False
    seed_ = True
    include_action_in_obs_ = False
    local_ = True
    Train_E2E_rl = TrainE2E(env_name=env_name_,
                            render=render_,
                            seed=seed_,
                            include_action_in_obs=include_action_in_obs_,
                            local=local_,
                            norm=False,
                            randomise=False,
                            )
    Train_E2E_rl.train(E2E=True, test_distance_metric=True, randomise=False)
    # Train_E2E_rl.play(50,
    #                   randomise=False,
    #                   withRotation=True,
    #                   withPause=False,
    #                   store_error=False,
    #                   collect_demonstration=False,
    #                   success_threshold=0.005,
    #                   test_distance_metric=True
    #                   )
