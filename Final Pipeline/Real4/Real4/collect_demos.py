from main import TrainEvaluateAgent
import pickle
import numpy as np
import csv
import time
import matplotlib.pyplot as plt
import rotations
import argparse
import os


class CollectDemos(TrainEvaluateAgent):
    def __init__(self, env_name, render, seed=True, diffusion=False, collect_demo=True, seed_idx=None):
        super().__init__(
            env_name_=env_name,
            render=render,
            real=False,
            display=False,
            diffusion=diffusion,
            seed=seed,
            collect_demo=collect_demo,
            seed_idx=seed_idx
        )
        self.step_size_limit = self.env.step_size_limit
        self.last_control_mode = -1

    def choose_slide_action_E2E(self, t, state_policyInput, desired_goal_policyInput, control_mode, pos_idx, real):
        """
        Choose action for sliding, and map to E2E continuous range
        """
        if t == 0:
            action, _, _ = self.agent.choose_action(state_policyInput,
                                                    desired_goal_policyInput,
                                                    train_mode=False)
            action = self.map_action_2_E2E(action)  # map to E2E range
            friction_state, control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])
        elif real:
            action, _, _, _, _ = \
                self.choose_action_with_filter_real(state_policyInput,
                                                    desired_goal_policyInput,
                                                    t,
                                                    control_mode,
                                                    pos_idx,
                                                    train_mode=False)
            action = self.map_action_2_E2E(action)  # map to E2E range
            friction_state, control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])
        else:
            action, _, _, _, _ = \
                self.choose_action_with_filter(state_policyInput,
                                               desired_goal_policyInput,
                                               self.r_dict,
                                               t,
                                               control_mode,
                                               train_mode=False)
            action = self.map_action_2_E2E(action)  # map to E2E range
            friction_state, control_mode, _ = self.discretize_action_to_control_mode_E2E(action[1])
        return action, friction_state, control_mode

    def step_multi(self, action, episode_dict, num_steps=11):
        for _ in range(num_steps):
            next_env_dict, r_dict, terminated, _, info_ = self.env.step(action)
            episode_dict['trajectory'].append(next_env_dict['observation'])
            r = r_dict["RL_IHM"]
            if terminated is True:
                # input("Press to continue")
                print("Terminated during the step")
            if r_dict["pos_control_position"] <= 0.03 \
                    or r_dict["torque_control_position"] >= 1.65 \
                    or terminated is True:
                break
        return next_env_dict, r_dict, terminated, info_, episode_dict

    def run_episode_rl_demos(self,
                             real=False,
                             reset=True,
                             withRotation=True,
                             evaluation=False,
                             randomisation=False,
                             repeat_same_goal=False
                             ):
        """
        Return not-normalized episode dict
        """
        print("Using RL policy for demo collection")
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
            'sampled_desired_goal': []
        }
        friction_change_times = 0
        per_success_rate = []
        episode_reward = 0
        print("| Random: ", randomisation)

        env_dict, reward_dict = self.reset(real=real, reset=reset, demo_collect_randomisation=randomisation, repeat_same_goal=repeat_same_goal)
        self.r_dict = reward_dict
        state, achieved_goal, desired_goal = self.extract_env_info(env_dict)
        if randomisation:
            # print("Randomisation: ", state[0], state[2])
            joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
            state[0] += joint_noise[0]
            state[2] += joint_noise[1]

        ''' Pick up if terminated '''
        inAir = False
        last_friction = 0
        control_mode = None
        pos_idx = 0
        terminated = False
        r = None
        t = None
        next_state = None
        next_env_dict = {}
        info_ = {}
        next_achieved_goal = None
        # self.change_side()
        # print(self.r_dict.keys())

        ''' Step '''
        for t in range(20):
            if evaluation:
                # desired_goal_policyInput = desired_goal[:-2] # TODO: uncomment this for VFF-bigSteps evaluation
                desired_goal_policyInput = desired_goal
                state_policyInput = state
            else:
                desired_goal_policyInput = self.r_dict['desired_goal_contact_point_radi']
                state_policyInput = np.concatenate([state[:8], self.r_dict['achieved_goal_contact_point_radi']])

            # print(state_policyInput)
            action, friction_state, control_mode = self.choose_slide_action_E2E(t,
                                                                                state_policyInput,
                                                                                desired_goal_policyInput,
                                                                                control_mode,
                                                                                pos_idx,
                                                                                real)
            # print(action)
            if real:
                raise ValueError("Not implemented")
            else:
                if friction_state != last_friction:
                    friction_change_times += 1
                    next_env_dict, r, terminated, _, info_ = self.friction_change(friction_state, self.env)
                    last_friction = friction_state

                if not terminated:
                    next_env_dict, r_dict, terminated, info_, episode_dict = self.step_multi(action, episode_dict)
                    self.r_dict = r_dict

            '''Store trajectories'''
            episode_dict["state"].append(np.array(state).copy())  # original obs contains radi info, don't want that
            episode_dict["action"].append(np.array(action.copy()))
            episode_dict["achieved_goal"].append(np.array(achieved_goal.copy()))
            # episode_dict['reward'].append(r.copy())
            episode_dict['terminals'].append(0)
            # print("Check_1", achieved_goal)

            next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(next_env_dict)
            # print("Achieved goal: ", next_achieved_goal)
            if randomisation:
                # print("Randomisation: ", next_state[0], next_state[2])
                joint_noise = self.domain_randomise.generate_gaussian_noise("joint_position", 2, correlated=False)
                next_state[0] += joint_noise[0]
                next_state[2] += joint_noise[1]
            state = next_state.copy()
            achieved_goal = next_achieved_goal.copy()
            desired_goal = next_desired_goal.copy()

            if real:
                raise ValueError("Not implemented")
            else:
                ''' Play '''
                per_success_rate.append(info_['is_success'])
                episode_reward += r
                if terminated is True:
                    print("Terminate: ", terminated)
                    break
                elif info_["is_success"] == 1:
                    print("Success: ", info_["is_success"])
                    break
                elif t > 10:
                    break

        ''' Rotation '''
        if real:
            raise ValueError("Not implemented")
        else:
            if withRotation is True and info_["is_success"] == 1:

                ''' Keep decelerating - doesn't Work '''
                # env_dict, self.r_dict, _, _, _ = self.friction_change_to_high(self.env)
                # control_mode = 5 / 6 + 1 / 12 * np.random.choice([1, -1])
                # if 5 / 6 > control_mode >= 4 / 6:
                #     pos_idx = 0
                # else:
                #     assert 1 >= control_mode >= 5 / 6, f"Wrong control mode, check: {control_mode}"
                #     pos_idx = 1
                # control_mode = control_mode * 2 - 1
                # action_range = env_dict['observation'][pos_idx * 2]
                # print("Pos idx: ", pos_idx, action_range)
                # # next_env_dict, r_dict, terminated, info_ = self.take_random_steps(control_mode, action_range,
                # #                                                                   episode_dict)
                # next_env_dict, r_dict, terminated, info_ = self.take_r`random_step(control_mode, episode_dict)
                # next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(next_env_dict)

                ''' Big step '''
                action = np.zeros(2)
                if withRotation is True and info_["is_success"] == 1:
                    rotation_precision = 0.003
                    _, self.r_dict, _, _, _ = self.friction_change_to_high(self.env)
                    rotation_action = np.array([self.r_dict['pos_control_position'], 0, False])
                    terminated, self.r_dict, env_dict, episode_dict = self.rotation.start_rotation(
                        rotation_action.copy(), rotation_precision, episode_dict)
                    next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(env_dict)
                    # TODO: Modify the rad range to policy range,
                    #  this is the range with unit rad instead of policy action as other steps,
                    #  remember to modify the set_action for: VFF-bigSteps
                    action[0] = abs(rotation_action[0] - self.r_dict["pos_control_position"])
                    action[1] = 5 / 6 - 1 / 12 + 1 / 6 * self.env.pos_idx
                    action[1] = action[1] * 2 - 1

                    episode_dict["state"].append(
                        np.array(state.copy()))  # original obs contains radi info, don't want that
                    episode_dict["achieved_goal"].append(np.array(achieved_goal.copy()))
                    episode_dict['terminals'].append(0)
                    episode_dict["action"].append(np.array(action.copy()))
                    # print('Check_2: ', achieved_goal)

                    if terminated:
                        print("\033[91m| Terminated during rotation \033[0m")
                    elif not terminated and self.r_dict['current_goal_centre_distance'] < rotation_precision:
                        print("\033[92m| Rotation Achieved with reward of: ",
                              self.r_dict['current_goal_centre_distance'], "\033[0m")
                        print("\033[92m| SUCCESS \033[0m")
                        print("\033[92m--------------------------------------------------------------------\033[0m")
                    else:
                        print("\033[91m| Rotation Failed with reward of: ", self.r_dict['current_goal_centre_distance'],
                              "\033[0m")
                        print("\033[91m| FAILED \033[0m")
                        print("\033[92m--------------------------------------------------------------------\033[0m")
                    # action, action_complete = self.rotation.reverse_rotate()
                    # print("check action: ", action)

        if real:
            raise ValueError("Not implemented")
        else:
            assert self.env_name == 'VariableFriction-v7', f"Current env is: {self.env_name}"
            assert len(next_achieved_goal) == 11, f"Check: {next_achieved_goal}"
            assert len(next_state) == 24, f"Check: {next_state}"

            last_achieved_goal = next_achieved_goal
            # print("Actual achieved goal:", last_achieved_goal)
            # print("Desired goal:", self.env.goal)
            episode_dict["desired_goal"] = [last_achieved_goal for _ in episode_dict["achieved_goal"]]
            episode_dict['sampled_desired_goal'] = [desired_goal for _ in episode_dict["achieved_goal"]]
            episode_dict["terminals"][-1] = 1
            episode_dict["next_state"] = episode_dict["state"][1:]
            episode_dict["next_state"].append(next_state)
            episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
            episode_dict["next_achieved_goal"].append(next_achieved_goal)
            episode_dict["next_state"].append(next_state)

            for idx, item in enumerate(episode_dict["state"]):
                pos_diff, angle_diff = self.compute_orientation_diff(np.array(episode_dict['desired_goal'][idx][:7]),
                                                              np.array(episode_dict['achieved_goal'][idx][:7]))
                radi_diff = episode_dict['desired_goal'][idx][7:9] - episode_dict['achieved_goal'][idx][7:9]
                episode_dict['state'][idx][-3] = radi_diff[0]
                episode_dict['state'][idx][-2] = radi_diff[1]
                episode_dict['state'][idx][-1] = angle_diff

            # print(np.shape(episode_dict["state"]))

        print(f"------------------------------------{t}")
        # per_success_rate.append(info_["is_success_old"])
        # print(info_["is_success_old"])
        # input("Press")
        # time.sleep(2)

        return episode_dict, per_success_rate, episode_reward, friction_change_times

    def collect_demos_with_rotation(self,
                                    num_episodes,
                                    real=False,
                                    withPause=False,
                                    keep_reset=True,
                                    demonstration_file_name=None,
                                    policy_path=None,
                                    test=False,
                                    rl_policy=True,
                                    display=False,
                                    withRotation=True,
                                    randomisation=False,
                                    goal_repeat_times=5
                                    ):

        demonstration_dict = {
            'observations': [],
            'desired_goals': [],
            'desired_goals_radi': [],
            'actions': [],
            'next_observations': [],
            'rewards': [],
            'terminals': [],
            'sampled_desired_goals': [],
        }

        if rl_policy:
            self.agent.load_weights_play(policy_path)
            self.agent.set_to_eval_mode()
        num_success = 0
        success = np.zeros(1)
        repeat_same_goal = False
        repeat_same_goal_count = 0
        ep = 0

        while True:
            if ep >= num_episodes-1:
                break

            print("episode: ", ep, ", Repeat: ", repeat_same_goal_count)
            reset = True if keep_reset or ep == 0 or success[-1] == 0 else False
            # reset = False

            if rl_policy:
                episode_dict, success, episode_reward, friction_change_times \
                    = self.run_episode_rl_demos(real=real, reset=reset, randomisation=randomisation, repeat_same_goal=repeat_same_goal)
            else:
                episode_dict, success, episode_reward, friction_change_times \
                    = self.run_episode_E2E_demo(real=real, reset=reset, display=display, withRotation=withRotation)

            status = "SUCCESS" if success[-1] else "Failed"
            num_success += success[-1]
            print(f"---- Episode {ep}/{num_episodes} {status} ----")
            if withPause:
                time.sleep(2)

            # print(np.shape(np.stack(episode_dict['state'])))
            if success[-1] == 0:
                # TODO: Check if filtering episodes that exceed certain length.
                print("Not save")

                # TODO: remove this for data collection
                # ep += 1
            else:
                if randomisation:
                    repeat_same_goal = True
                    repeat_same_goal_count += 1

                    if repeat_same_goal_count >= goal_repeat_times:
                        repeat_same_goal = False
                        repeat_same_goal_count = 0
                        ep += 1

                else:
                    ep += 1

                demonstration_dict['observations'].append(np.stack(episode_dict['state']))
                demonstration_dict['next_observations'].append(np.stack(episode_dict['next_state']))
                demonstration_dict['desired_goals'].append(np.stack(episode_dict['desired_goal']))
                demonstration_dict['actions'].append(np.stack(episode_dict['action']))
                demonstration_dict['rewards'].append(episode_dict['reward'])
                demonstration_dict['terminals'].append(episode_dict['terminals'])
                demonstration_dict['sampled_desired_goals'].append(episode_dict['sampled_desired_goal'])

            # folder_path = 'trajectory_real'
            # if not os.path.exists(folder_path):
            #     os.makedirs(folder_path)
            # file_path = os.path.join(folder_path, f'{demonstration_file_name}_{ep}.pkl')
            # with open(file_path, 'wb') as file:
            #     pickle.dump(episode_dict, file)
            # print(f"Data has been serialized to {file_path}")

            if (ep + 1) % 10 == 0:
                print(f"| {demonstration_file_name}")
                print("| Success rate: ", num_success / (ep + 1))
                print("| Success threshold: ", self.env.r_threshold)
                if not test:
                    self.store_demonstration(demonstration_dict, demonstration_file_name)
        if not test:
            self.store_demonstration(demonstration_dict, demonstration_file_name)
        self.env.close()

    def map_action_2_E2E(self, action):
        '''
        Map action to E2E discretization region, it was -1 to 1,
        now it should be 0 to 4/6,
        which corresponds to -1 to 2*4/6-1
        '''
        action_e2e = action
        action_norm = (action[1] + 1) / 2  # map from -1 ~ 1 to 0 ~ 1
        action_norm_e2e = action_norm * 4 / 6  # map from 0 ~ 1 to 0 ~ 4/6
        action_e2e[1] = action_norm_e2e * 2 - 1
        return action_e2e

    def store_demonstration(self, demonstration_dict, demonstration_file_name):
        demonstrations_dict = {
            'observations': np.vstack(demonstration_dict['observations']).astype(np.float32),
            'next_observations': np.vstack(demonstration_dict['next_observations']).astype(np.float32),
            'desired_goals': np.vstack(demonstration_dict['desired_goals']).astype(np.float32),
            'sampled_desired_goals': np.vstack(demonstration_dict['sampled_desired_goals']).astype(np.float32),
            'actions': np.vstack(demonstration_dict['actions']).astype(np.float32),
            'rewards': np.hstack(demonstration_dict['rewards']).astype(np.float32),
            'terminals': np.hstack(demonstration_dict['terminals']).astype(np.float32),
        }
        assert demonstration_file_name is not None, f"File name is None, check: {demonstration_file_name}"
        self.save_as_pickle(demonstrations_dict, demonstration_file_name)

    def take_random_steps(self, control_mode, max_step_size, episode_dict, num_of_steps=5):
        """
        Take random number of steps of rotation
        """
        step_size = max_step_size
        # print(num_of_steps, max_step_size)
        # policy -> env: (action + 1) / 2
        for i in range(num_of_steps):
            step_size_scale = np.random.uniform(0, 1)
            step_size *= step_size_scale
            step_size_env = np.clip(step_size / 1.8807 * 2 - 1, -1, 1)  # convert to policy range
            action = [step_size_env, control_mode]
            next_env_dict, r_dict, terminated, info_ = self.step_multi(action)
            next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(next_env_dict)

            if i > 0:
                episode_dict["state"].append(np.array(state).copy())  # original obs contains radi info, don't want that
                episode_dict["action"].append(np.array(action.copy()))
                episode_dict["achieved_goal"].append(np.array(achieved_goal.copy()))
                episode_dict['terminals'].append(False)

            state = next_state.copy()
            achieved_goal = next_achieved_goal.copy()
            desired_goal = next_desired_goal.copy()
        return next_env_dict, r_dict, terminated, info_

    def take_random_step(self, control_mode, episode_dict):
        """
        Take single step with random amount
        """
        # step_size = max_step_size
        # print(num_of_steps, max_step_size)
        # policy -> env: (action + 1) / 2
        num_of_steps = np.random.randint(1, 3)
        for i in range(num_of_steps):
            step_size = np.random.uniform(-1, 1)
            action = [step_size, control_mode]
            print(i, action)
            next_env_dict, r_dict, terminated, info_ = self.step_multi(action)
            next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(next_env_dict)

            if i > 0:
                episode_dict["state"].append(np.array(state).copy())  # original obs contains radi info, don't want that
                episode_dict["action"].append(np.array(action.copy()))
                episode_dict["achieved_goal"].append(np.array(achieved_goal.copy()))
                episode_dict['terminals'].append(False)

            state = next_state.copy()
            achieved_goal = next_achieved_goal.copy()
            desired_goal = next_desired_goal.copy()
        return next_env_dict, r_dict, terminated, info_

    def slide(self):
        """
        Step size decreases to either 0 or a value, the next slide should start from the ending step size
        the amount of slide should be random to cover enough range (very samll ~ max limit)
        randomly select the decreasing ratio (deceleration) and same action
        randomly select a set of control mode, 2 or 3
        """
        pass

    def select_control_mode(self, slide):
        while True:
            if slide:
                control_mode = np.random.randint(0, 4)  # randomly select between 0 ~ 4
            else:
                ''' rotation '''
                control_mode = np.random.randint(4, 6)  # randomly select between 0 ~ 4
            if control_mode != self.last_control_mode:
                self.last_control_mode = control_mode
                break
        control_mode_con = self.normalize_control_mode(control_mode)  # from 0,1,2,3,4,5 to 0, 1/6, 2/6 ... 5/6 to -1, 1
        friction_state, control_mode_dis, pos_idx = self.discretize_action_to_control_mode_E2E(control_mode_con)
        assert control_mode_dis == control_mode, f"Wrong mapping, check: {control_mode_dis, control_mode, control_mode_con}"
        return control_mode_con, friction_state, control_mode_dis, pos_idx

    def select_control_mode_with_filter(self, state, slide):
        action_discrete, friction_state, control_mode, pos_idx = self.select_control_mode(slide=slide)
        ''' If the leading finger is reaches the limit'''
        if (state[pos_idx * 2] <= 0.03 or state[pos_idx * 2] >= 1.65
                or state[(1 - pos_idx) * 2] <= 0 or state[(1 - pos_idx) * 2] >= 1.65):
            last_pos_idx = pos_idx
            while True:
                action_discrete, friction_state, control_mode, pos_idx = self.select_control_mode(slide=slide)
                if last_pos_idx != pos_idx:
                    return action_discrete, friction_state, control_mode, pos_idx
        return action_discrete, friction_state, control_mode, pos_idx

    def get_next_position(self, current_movement, goal_movement, current_movement_radi, current_movement_radi_limit,
                          slide, step_size_lower_limit=0.005):
        relative_action = np.sin((current_movement / goal_movement) * np.pi) * self.step_size_limit
        if slide:
            relative_action_radi = np.sin(
                (current_movement_radi / current_movement_radi_limit) * np.pi) * self.step_size_limit
            relative_action_clip = np.clip(min(relative_action, relative_action_radi), step_size_lower_limit,
                                           self.step_size_limit)
            if current_movement_radi > current_movement_radi_limit:
                relative_action_clip = 0
        else:
            relative_action_clip = np.clip(relative_action, step_size_lower_limit, self.step_size_limit)
        relative_action_policy = self.rad_2_policy(relative_action_clip)
        return relative_action_policy, relative_action_clip

    def get_next_position_1(self, current_position, total_movement, num_steps, current_step):
        """
        Calculate the next position in a smooth trajectory with slow acceleration and deceleration.

        Args:
        - current_position: Current position in the trajectory.
        - goal_position: The final goal position.
        - num_steps: Total number of steps to reach the goal.
        - current_step: The current step number.

        Returns:
        - next_position: The next position in the trajectory.
        """

        half_steps = num_steps // 2

        if current_step < half_steps:
            # Acceleration phase
            t = current_step / half_steps
            movement_factor = 0.5 * (1 - np.cos(np.pi * t))  # Cosine function for smooth acceleration
        else:
            # Deceleration phase
            t = (current_step - half_steps) / half_steps
            movement_factor = 0.5 * (1 + np.cos(np.pi * t))  # Cosine function for smooth deceleration

        action_rad = np.clip(movement_factor * (total_movement / num_steps), 0, self.step_size_limit)
        action_policy = self.rad_2_policy(action_rad)
        return action_policy, action_rad

    def rad_2_policy(self, action_rad):
        action_policy = (action_rad / self.step_size_limit) * 2 - 1
        assert -1 <= action_policy <= 1, f"Wrong action size: {action_policy, action_rad}"
        return action_policy

    def take_move(self, episode_dict, env_dict, slide, last_friction):
        """
        Take 10 steps
        state: 10 (1+9)
        actions: 10 (10)
        achieved_goal: 10 (1+9)
        terminals: 10 (10)
        """
        action = np.zeros(2)  # (action step size, discrete control mode in 0~1)
        state, achieved_goal, desired_goal = self.extract_env_info(env_dict)
        action[1], friction_state, control_mode, pos_idx = self.select_control_mode_with_filter(state, slide=slide)
        radi_idx = np.clip(friction_state, 0, 1) - 3
        goal_movement = np.random.uniform(0.03, state[pos_idx * 2])
        current_movement = 0
        current_movement_radi = 0
        slide_r_limit = np.array(self.env.terminate_r_limit) + np.array([0.005, -0.005])
        if control_mode == 0 or control_mode == 2:  # slide up
            current_movement_radi_limit = slide_r_limit[1] - state[radi_idx]
        else:  # slide down or rotate
            current_movement_radi_limit = state[radi_idx] - slide_r_limit[0]
        next_env_dict = {}
        per_success_rate = []
        terminated = False

        if friction_state != last_friction:
            next_env_dict, r, terminated, _, info_ = self.friction_change(friction_state, self.env)

        print("Goal: ", goal_movement)
        while True:
            # TODO: Uncomment this for trajectory: accelerate + decelerate
            action[0], action_rad = self.get_next_position(current_movement,
                                                           goal_movement,
                                                           current_movement_radi,
                                                           current_movement_radi_limit,
                                                           slide)

            # TODO: Uncomment this for trajectory: fixed step size
            # action[0] = 0.01 / 1.8807

            # # TODO: Uncomment this for trajectory: fixed step size
            # action[0] = goal_movement

            next_env_dict, r_dict, terminated, _, info_ = self.env.step(action)
            episode_dict['state'].append(state.copy())
            episode_dict['action'].append(action.copy())
            episode_dict['achieved_goal'].append(np.array(achieved_goal.copy()))
            episode_dict['terminals'].append(False)
            per_success_rate.append(0)

            next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(next_env_dict)
            current_movement += abs(next_state[pos_idx * 2] - state[pos_idx * 2])  # moved amount
            current_movement_radi += abs(next_state[radi_idx] - state[radi_idx])
            state = next_state.copy()
            achieved_goal = next_achieved_goal.copy()

            if (current_movement - goal_movement > 0.005 or
                    (current_movement_radi > current_movement_radi_limit and slide is True) or
                    len(per_success_rate) > 500):
                break
        time.sleep(2)
        return episode_dict, next_env_dict, per_success_rate, friction_state, pos_idx

    def take_move_con(self, episode_dict, env_dict, slide, last_friction):
        """
        Take 10 steps
        state: 10 (1+9)
        actions: 10 (10)
        achieved_goal: 10 (1+9)
        terminals: 10 (10)
        """
        action = np.zeros(2)  # (action step size, discrete control mode in 0~1)
        state, achieved_goal, desired_goal = self.extract_env_info(env_dict)
        action[1], friction_state, control_mode, pos_idx = self.select_control_mode_with_filter(state, slide=slide)
        radi_idx = np.clip(friction_state, 0, 1) - 3
        if slide:
            goal_movement = np.random.uniform(0.03, state[pos_idx * 2]) / 2
        else:
            goal_movement = np.random.uniform(0.03, state[pos_idx * 2])
        per_success_rate = []

        if friction_state != last_friction:
            next_env_dict, r, terminated, _, info_ = self.friction_change(friction_state, self.env)
            state, achieved_goal, desired_goal = self.extract_env_info(next_env_dict)

        action[0] = (goal_movement / 1.8807) * 2 - 1

        while True:
            # TODO: Uncomment this for random policy with big steps
            next_env_dict, r_dict, terminated, _, info_ = self.env.step(action)

            if len(per_success_rate) == 0:
                start_pos = state[self.env.pos_idx * 2]
                # print("Start: ", start_pos)
                # print("Start pos: ", start_pos)

            next_state, next_achieved_goal, next_desired_goal = self.extract_env_info(next_env_dict)
            per_success_rate.append(0)

            if terminated is True and slide is True:
                print("\033[91m | Terminate ---- Don't Save the rest \033[0m")
                # print("Check")
                break
            elif len(per_success_rate) > 200:
                print("\033[91m | Stuck ---- Don't Save the rest \033[0m")
                break
            elif abs(abs(start_pos - next_state[self.env.pos_idx * 2]) - goal_movement) < 0.005:
                # print("Actual Moved: ", start_pos - next_state[self.env.pos_idx * 2], next_state[self.env.pos_idx * 2])
                if slide is False:
                    print("| Rotation complete")
                per_success_rate.append(1)
                episode_dict['state'].append(state.copy())
                episode_dict['achieved_goal'].append(np.array(achieved_goal.copy()))
                episode_dict['terminals'].append(False)
                episode_dict['action'].append(action.copy())
                break

        # print(" --------------------------------------------------------------------------- ")
        # input(f"Press to proceed, pos_idx: {self.env.pos_idx}")

        return episode_dict, next_env_dict, per_success_rate, friction_state, pos_idx

    def run_episode_E2E_small_steps_demo(self, real, reset, withRotation=True, display=False):
        """
        Move (slide and rotation) characteristics:
        1. Decreasing step size until zero for every move
        2. Step size within each slide upper limit = 1.8807 / 7
        3. The amount of slide should be random to cover enough range (very samll ~ max limit)
        4. Single steps
        5. Remain the step size if exceed the upper limit

        Plan:
        1. Select the scaling ratio < 1, to keep decreasing the step size

        """
        episode_dict = {
            "state": [],
            "action": [],
            "achieved_goal": [],
            "next_achieved_goal": [],
            "desired_goal": [],
            "next_state": [],
            "terminals": [],
            "reward": [],
            "sampled_desired_goal": []
        }
        next_env_dict, reward_dict = self.reset(real=real, reset=reset)
        state, achieved_goal, desired_goal = self.extract_env_info(next_env_dict)
        # print("| Desired goal: ", desired_goal)
        num_steps = np.random.randint(4, 6)  # number of action types in single episode
        per_success_rate = []
        episode_reward = None
        friction_change_times = None
        friction_state = None
        pos_idx = None
        if withRotation:
            with_rotation = np.random.random() < 0.75
            if with_rotation:
                print("| With Rotation")
        else:
            with_rotation = False
            print("| Without Rotation")
        print(f"| Number of steps: {num_steps - 1 + int(with_rotation)}")
        for i in range(num_steps):
            if i < num_steps - 1:
                ''' slide '''
                episode_dict, next_env_dict, per_success_rate, friction_state, pos_idx \
                    = self.take_move(episode_dict,
                                     next_env_dict,
                                     slide=True,
                                     last_friction=friction_state,
                                     )
            elif i == num_steps - 1:
                ''' rotation '''
                if with_rotation is True:
                    episode_dict, next_env_dict, per_success_rate, friction_state, pos_idx \
                        = self.take_move(episode_dict,
                                         next_env_dict,
                                         slide=False,
                                         last_friction=friction_state,
                                         )
                else:
                    pass
                episode_dict[-1] = True
            else:
                ''' change side '''
                raise ValueError("Change side is not implemented")

        # input("Press to continue")

        if len(per_success_rate) >= 500:
            print("\033[91m | Not Save \033[0m")
            pass
        else:
            print("\033[92m | Saved \033[0m")
            state, achieved_goal, desired_goal = self.extract_env_info(next_env_dict)
            # print("| Desired goal: ", desired_goal)
            last_achieved_goal = achieved_goal
            per_success_rate[-1] = 1
            episode_dict["terminals"][-1] = True
            episode_dict["desired_goal"] = [last_achieved_goal for _ in episode_dict["achieved_goal"]]
            episode_dict['sampled_desired_goal'] = [desired_goal for _ in episode_dict["achieved_goal"]]
            episode_dict["next_state"] = episode_dict["state"][1:]
            episode_dict["next_state"].append(state.copy())
            episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
            episode_dict["next_achieved_goal"].append(achieved_goal)

            for idx, item in enumerate(episode_dict["state"]):
                _, angle_diff = self.compute_orientation_diff(np.array(episode_dict['desired_goal'][idx][:7]),
                                                              np.array(episode_dict['achieved_goal'][idx][:7]))
                radi_diff = episode_dict['desired_goal'][idx][7:9] - episode_dict['achieved_goal'][idx][7:9]
                episode_dict['state'][idx][-3] = radi_diff[0]
                episode_dict['state'][idx][-2] = radi_diff[1]
                episode_dict['state'][idx][-1] = angle_diff

        "Plot"
        if display:
            fig, axs = plt.subplots(4, 1, figsize=(10, 12))
            # Plot the action in the first subplot
            axs[0].plot((np.array(episode_dict['action'])[:, 0] + 1) / 2, marker='o')
            axs[0].set_xlabel('Steps')
            axs[0].set_ylabel('Action')
            axs[0].set_title('Action Trajectory')
            axs[0].grid(True)
            # Plot the state in the second subplot
            axs[1].plot(np.array(episode_dict['state'])[:, -2], marker='o')
            axs[1].set_xlabel('Steps')
            axs[1].set_ylabel('State')
            axs[1].set_title('State Trajectory')
            axs[1].grid(True)

            axs[2].plot(np.array(episode_dict['state'])[:, -3], marker='o')
            axs[2].set_xlabel('Steps')
            axs[2].set_ylabel('State')
            axs[2].set_title('State Trajectory')
            axs[2].grid(True)

            axs[3].plot(np.array(episode_dict['state'])[:, -1], marker='o')
            axs[3].set_xlabel('Steps')
            axs[3].set_ylabel('State')
            axs[3].set_title('State Trajectory')
            axs[3].grid(True)

            fig.suptitle('Real-time Trajectory with Slow Acceleration and Slow Deceleration', fontsize=16)
            plt.show()

        return episode_dict, per_success_rate, episode_reward, friction_change_times

    def run_episode_E2E_demo(self, real, reset, withRotation=True, display=False):
        """
        Move (slide and rotation) characteristics:
        1. Decreasing step size until zero for every move
        2. Step size within each slide upper limit = 1.8807 / 7
        3. The amount of slide should be random to cover enough range (very samll ~ max limit)
        4. Single steps
        5. Remain the step size if exceed the upper limit

        Plan:
        1. Select the scaling ratio < 1, to keep decreasing the step size

        """
        episode_dict = {
            "state": [],
            "action": [],
            "achieved_goal": [],
            "next_achieved_goal": [],
            "desired_goal": [],
            "next_state": [],
            "terminals": [],
            "reward": [],
            # "sampled_desired_goal": []
        }
        next_env_dict, reward_dict = self.reset(real=real, reset=reset)
        state, achieved_goal, desired_goal = self.extract_env_info(next_env_dict)
        # print("| Desired goal: ", desired_goal)
        num_steps = np.random.randint(4, 6)  # number of action types in single episode
        per_success_rate = []
        episode_reward = None
        friction_change_times = None
        friction_state = None
        pos_idx = None
        if withRotation:
            with_rotation = np.random.random() < 0.75
            if with_rotation:
                print("| With Rotation")
        else:
            with_rotation = False
            print("| Without Rotation")
        print(f"| Number of steps: {num_steps - 1 + int(with_rotation)}")
        for i in range(num_steps):
            if i < num_steps - 1:
                ''' slide '''
                episode_dict, next_env_dict, per_success_rate, friction_state, pos_idx \
                    = self.take_move_con(episode_dict,
                                         next_env_dict,
                                         slide=True,
                                         last_friction=friction_state,
                                         )
            elif i == num_steps - 1:
                ''' rotation '''
                if with_rotation is True:
                    episode_dict, next_env_dict, per_success_rate, friction_state, pos_idx \
                        = self.take_move_con(episode_dict,
                                             next_env_dict,
                                             slide=False,
                                             last_friction=friction_state,
                                             )
                else:
                    pass
                episode_dict[-1] = True
            else:
                ''' change side '''
                raise ValueError("Change side is not implemented")

        # input("Press to continue")

        if len(per_success_rate) >= 500:
            print("\033[91m | Not Save \033[0m")
            pass

        else:
            print("\033[92m | Saved \033[0m")
            print(f"\033[92m | Episode length: {len(episode_dict['terminals'])} \033[0m")
            state, achieved_goal, desired_goal = self.extract_env_info(next_env_dict)
            last_achieved_goal = achieved_goal[:9]
            per_success_rate[-1] = 1
            episode_dict["terminals"][-1] = True
            episode_dict["desired_goal"] = [last_achieved_goal for _ in episode_dict["achieved_goal"]]
            # episode_dict['sampled_desired_goal'] = [desired_goal for _ in episode_dict["achieved_goal"]]
            episode_dict["next_state"] = episode_dict["state"][1:]
            episode_dict["next_state"].append(state.copy())
            episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
            episode_dict["next_achieved_goal"].append(achieved_goal)

            for key in episode_dict.keys():
                print(key, np.shape(episode_dict[key]))

            for idx, item in enumerate(episode_dict["state"]):
                _, angle_diff = self.compute_orientation_diff(np.array(episode_dict['desired_goal'][idx][:7]),
                                                              np.array(episode_dict['achieved_goal'][idx][:7]))
                radi_diff = episode_dict['desired_goal'][idx][7:9] - episode_dict['achieved_goal'][idx][7:9]
                episode_dict['state'][idx][-3] = radi_diff[0]
                episode_dict['state'][idx][-2] = radi_diff[1]
                episode_dict['state'][idx][-1] = angle_diff

        "Plot"
        if display:
            fig, axs = plt.subplots(4, 1, figsize=(10, 12))
            # Plot the action in the first subplot
            axs[0].plot((np.array(episode_dict['action'])[:, 0] + 1) / 2, marker='o')
            axs[0].set_xlabel('Steps')
            axs[0].set_ylabel('Action')
            axs[0].set_title('Action Trajectory')
            axs[0].grid(True)
            # Plot the state in the second subplot
            axs[1].plot(np.array(episode_dict['state'])[:, -2], marker='o')
            axs[1].set_xlabel('Steps')
            axs[1].set_ylabel('State')
            axs[1].set_title('State Trajectory')
            axs[1].grid(True)

            axs[2].plot(np.array(episode_dict['state'])[:, -3], marker='o')
            axs[2].set_xlabel('Steps')
            axs[2].set_ylabel('State')
            axs[2].set_title('State Trajectory')
            axs[2].grid(True)

            axs[3].plot(np.array(episode_dict['state'])[:, -1], marker='o')
            axs[3].set_xlabel('Steps')
            axs[3].set_ylabel('State')
            axs[3].set_title('State Trajectory')
            axs[3].grid(True)

            fig.suptitle('Real-time Trajectory with Slow Acceleration and Slow Deceleration', fontsize=16)
            plt.show()

        return episode_dict, per_success_rate, episode_reward, friction_change_times

    def change_side(self):
        pass

    @staticmethod
    def normalize_control_mode(control_mode):
        return (control_mode / 6 + 1 / 12) * 2 - 1

    def compute_orientation_diff(self, goal_a, goal_b):
        ''' get pos difference and rotation difference
        left motor pos: 0.037012 -0.1845 0.002
        right motor pos: -0.037488 -0.1845 0.002
        '''
        assert goal_a.shape == goal_b.shape, f"Check: {goal_a.shape}, {goal_b.shape}"
        assert goal_a.shape[-1] == 7
        goal_a[2] = goal_b[2]

        d_pos = np.zeros_like(goal_a[..., 0])

        delta_pos = goal_a[..., :3] - goal_b[..., :3]
        d_pos = np.linalg.norm(delta_pos, axis=-1)

        quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]

        euler_a = rotations.quat2euler(quat_a)
        euler_b = rotations.quat2euler(quat_b)
        if euler_a.ndim == 1:
            euler_a = euler_a[np.newaxis, :]  # Reshape 1D to 2D (1, 3)
        if euler_b.ndim == 1:
            euler_b = euler_b[np.newaxis, :]  # Reshape 1D to 2D (1, 3)
        euler_a[:, :2] = euler_b[:, :2]  # make the second and third term of euler angle the same
        quat_a = rotations.euler2quat(euler_a)
        quat_a = quat_a.reshape(quat_b.shape)

        # print(quat_a, quat_b)
        quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))  # q_diff = q1 * q2*
        angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1.0, 1.0))
        d_rot = angle_diff
        return d_pos, d_rot

    def ihm_step(self, env, action, last_friction_state):
        friction_state, control_mode_dis, pos_idx = self.discretize_action_to_control_mode_E2E(action[0])
        if last_friction_state == friction_state:
            obs, r_dict, terminated, _, info_ = env.step(action)
        else:
            obs, r_dict, terminated, _, info_ = self.friction_change(friction_state, self.env)
            if not terminated:
                obs, r_dict, terminated, _, info_ = env.step(action)

        _, angle_diff = self.compute_orientation_diff(np.array(obs['desired_goal'][:7]),
                                                      np.array(obs['achieved_goal'][:7]))
        radi_diff = obs['desired_goal'][7:9] - obs['achieved_goal'][7:9]
        obs['observation'][-1] = angle_diff
        obs['observation'][-2] = radi_diff[1]
        obs['observation'][-3] = radi_diff[0]
        return obs, r_dict, terminated, terminated, info_, friction_state

    def test(self):
        action = [0.5, 0.5]
        last_friction_state = 0
        for i in range(200):
            print(i)
            obs_dict, reward, done, _, info, friction_state = self.ihm_step(self.env, action, last_friction_state)


# if __name__ == "__main__":
#     env_name_ = "VariableFriction-v8"
#     render_ = False
#     withPause_ = False
#     display_ = False
#
#     demonstration_file_name_ = "VFF-random"
#     policy_path_ = '/Users/qiyangyan/Desktop/Training Files/Trained Policy/Training4_2mm_DR/VariableFriction_3_24.pth'
#
#     demo_collection = CollectDemos(env_name=env_name_,
#                                    render=render_
#                                    )
#     demo_collection.collect_demos_with_rotation(10000,
#                                                 real=False,
#                                                 withPause=withPause_,
#                                                 demonstration_file_name=demonstration_file_name_,
#                                                 policy_path=policy_path_,
#                                                 test=False,
#                                                 rl_policy=False,
#                                                 display=display_,
#                                                 withRotation=True
#                                                 )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect demonstrations for a specified environment.")
    parser.add_argument('--env_name', type=str, default="VariableFriction-v8", help='Name of the environment')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--randomise', action='store_true', help='Randomise the environment')
    parser.add_argument('--not_seed', action='store_false', help='Not seed the environment')
    parser.add_argument('--withPause', action='store_true', help='Add pause between steps')
    parser.add_argument('--display', action='store_true', help='Display each episode of collected demonstrations')
    parser.add_argument('--demonstration_file_name', type=str, default="VFF-test",
                        help='Name of the demonstration file')
    parser.add_argument('--policy_path', type=str,
                        default='/Users/qiyangyan/Desktop/TrainingFiles/Trained '
                                'Policy/Training4_2mm_DR/VariableFriction_3_24.pth',
                        help='Path to the trained policy file')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--withoutRotation', action='store_false', help='With rotation at the episode end or not')
    parser.add_argument("--diffusion", action='store_true', help="Use diffusion to collect demonstration")
    parser.add_argument("--seed_idx", type=int, default=None, help='Seed to seed the environment')
    args = parser.parse_args()

    print("With Rotation: ", args.withoutRotation)

    # TODO: VariableFriction-v7 + RL + withRotation: python3 collect_demos.py --env_name "VariableFriction-v7" --episodes 2000 --demonstration_file_name "VFF-bigSteps"
    #   1. Modify the robot to change is_success in info
    # TODO:
    #  Command: python3 collect_demos.py --env_name "VariableFriction-v7" --episodes 2000 --demonstration_file_name "test" --render --policy_path "/Users/qiyangyan/Desktop/TrainingFiles/Trained Policy/Training3_5mm_DR/VariableFriction.pth"
    demo_collection = CollectDemos(env_name=args.env_name,
                                   render=args.render,
                                   seed=args.not_seed,
                                   diffusion=args.diffusion,
                                   collect_demo=True,
                                   seed_idx=args.seed_idx,
                                   )
    print("rl path: ", args.policy_path)
    demo_collection.collect_demos_with_rotation(num_episodes=args.episodes,
                                                real=False,
                                                withPause=args.withPause,
                                                demonstration_file_name=args.demonstration_file_name,
                                                policy_path=args.policy_path,
                                                test=True,
                                                rl_policy=True,  # TODO: Modify this (rl_policy is True when use rl policy to collect demos)
                                                display=args.display,
                                                withRotation=args.withoutRotation,
                                                randomisation=args.randomise
                                                )
    # demo_collection.test()
