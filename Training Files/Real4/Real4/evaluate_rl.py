from main import TrainEvaluateAgent
from collect_demos import CollectDemos
import pickle
import numpy as np
import csv
import time
import os
import rotations
from scipy.spatial.transform import Rotation as R
from agents.simple_nn import SimpleNN, EnhancedNN, TransformerTabNet
import argparse


class EvaluateDiffusion(TrainEvaluateAgent):
    def __init__(self, env_name, render, dataset, output_dir, model_idx, seed, layer_dim, include_action_in_obs,
                 separate_policies, terminate_indicator=False, terminate_model_name=None, termiante_save_path=None,
                 control_mode_indicator=False):
        super().__init__(
            env_name_=env_name,
            render=render,
            real=False,
            display=False,
            diffusion=True,
            seed=seed,
            dataset=dataset,
            output_dir=output_dir,
            model_idx=model_idx,
            layer_dim=layer_dim,
            include_action_in_obs=include_action_in_obs,
            separate_policies=separate_policies,  # Use separate policies for sliding and rotation
            terminate_indicator=terminate_indicator,
            terminate_model_name=terminate_model_name,
            termiante_save_path=termiante_save_path,
            control_mode_indicator=control_mode_indicator,
        )

    def _goal_distance(self, goal_a, goal_b):
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
        # assert d_pos.shape == d_rot.shape
        return d_pos, d_rot
        # return d_pos

    def convert_quat_to_euler(self, pose_quat):
        quat = pose_quat[3:]
        rotation = R.from_quat(quat)
        euler = rotation.as_euler('xyz', degrees=True)
        pose_euler = np.concatenate([pose_quat[:3], euler])
        return pose_euler

    def eval_diffusion_with_rotation(self,
                                     num_episodes,
                                     real=False,
                                     withPause=False,
                                     keep_reset=True,
                                     demonstration_file_name=None,
                                     policy_path=None,
                                     num_steps=None,
                                     store_trajectory=False,
                                     h1=1,
                                     h2=1,
                                     h3=1,
                                     normalise=False,
                                     data_4_norm='/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10000demos_slide'
                                     ):

        self.agent.load_weights_play(policy_path)
        self.agent.set_to_eval_mode()
        num_success = 0
        num_success_slide = 0
        success = np.zeros(1)

        pos_error_list = []
        orientation_error_list = []

        for ep in range(num_episodes):
            print("episode: ", ep, num_episodes)
            reset = True if keep_reset or ep == 0 or success[-1] == 0 else False

            if h1 == 1 and h2 == 1 and h3 == 1:
                episode_dict, success, _, _, _, slide_succeed \
                    = self.run_episode_E2E(real=real, reset=reset, train_mode=False, normalise=normalise)
            else:
                print("Stack")
                episode_dict, success, _, _, _ \
                    = self.run_episode_E2E_stack(real=real, reset=reset, train_mode=False, num_steps=num_steps,
                                                 pred_horizon=h1, obs_horizon=h2, action_horizon=h3)

            status = "SUCCESS" if success[-1] == 1 else "Failed"
            num_success += success[-1]
            print(f"---- Episode {ep + 1} {status} ----")
            print(num_success)
            if slide_succeed is True:
                num_success_slide += 1
            if withPause:
                time.sleep(2)

            if success[-1] == 0:
                print("Not save")
            else:
                pos_error, orientation_error = self._goal_distance(np.array(episode_dict['desired_goal'][-1][:7]),
                                                                   np.array(episode_dict['achieved_goal'][-1][:7]))
                pos_error_list.append(pos_error)
                orientation_error_list.append(orientation_error)

            if (ep + 1) % 10 == 0:
                print(f"| {demonstration_file_name}")
                print("| Success rate: ", num_success / (ep + 1.0))
                print("| Slide Success rate: ", num_success_slide / (ep + 1.0))
                print("| Success threshold: ", self.env.r_threshold)
                print(
                    f"| Pos error statics: mean = {np.mean(np.array(pos_error_list))}, std = {np.std(np.array(pos_error_list))}")
                print(
                    f"| Orientation error statics: mean = {np.mean(np.array(orientation_error_list))}, std = {np.std(np.array(orientation_error_list))}")

            if store_trajectory:
                folder_path = demonstration_file_name
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                file_path = os.path.join(folder_path, f'{demonstration_file_name}_{ep}.pkl')
                with open(file_path, 'wb') as file:
                    pickle.dump(episode_dict, file)
                print(f"Data has been serialized to {file_path}")

        self.env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect demonstrations for a specified environment.")
    parser.add_argument('--env_name', type=str, default="VariableFriction-v7", help='Name of the environment')
    parser.add_argument('--output_dir', type=str, help='Policy Path')
    parser.add_argument('--datapath', type=str, help='Demonstration Path')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--seed', action='store_true', help='Seed the environment')
    parser.add_argument('--include_action_in_obs', action='store_true', help='Include action in observation')
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of episodes to evaluate')
    parser.add_argument('--model_idx', type=int, default=20000, help='Actor_num')
    parser.add_argument('--layer_dim', type=int, default=1024, help='Dimension of layer')
    parser.add_argument('--h1', type=int, default=1, help='Predict horizon')
    parser.add_argument('--h2', type=int, default=1, help='Obs horizon')
    parser.add_argument('--h3', type=int, default=1, help='Action horizon')
    parser.add_argument('--eval_task_selection', type=int, default=0, help='Choose a task directly without specifying '
                                                                           'parameters')
    parser.add_argument('--terminate_indicator', action='store_true', help='Terminate episode with trained policy')
    parser.add_argument('--control_mode_indicator', action='store_true', help='Control mode with classifier')
    args = parser.parse_args()

    print(args)

    ' Default Settings '
    # render_ = False
    # num_episodes_ = 500
    # include_action_in_obs = False
    # seed = True

    # output_dir_ = "/Users/qiyangyan/Desktop/Diffusion/Policies/models_18_1024"

    ' Random policy '
    # env_name_ = "VariableFriction-v8"  # random policy

    ' RL policy collected demos '
    # env_name_ = "VariableFriction-v7"  # RL policy with rotation

    # TODO: Uncomment below for the task evaluation
    #   1. Big Steps
    #   2. Big Steps with action chunking
    ' -1- '
    # # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_19_1024'
    # # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_20_1024'
    # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_21_1024'
    # model_idx_ = 10000
    # layer_dim_ = 1024
    # with open(f'/Users/qiyangyan/Desktop/Training Files/Real4/Real4/demonstration/VFF-bigSteps-1879demos-2mmPolicy-5mmThreshold', 'rb') as f:
    #     dataset = pickle.load(f)
    # dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])
    # for item in dataset.keys():
    #     print(np.shape(dataset[item]))

    ' -2- '
    # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_22_1024_2000'
    # model_idx_ = 20000
    # layer_dim_ = 1024
    # h1 = 4
    # h2 = 2
    # h3 = 2
    # datapath = '/Users/qiyangyan/Desktop/TrainingFiles/Diffusion1/get_stacked_dataset/VFF-bigSteps-2000demos_stacked_4_2_2.pkl'
    # with open(datapath, 'rb') as f:
    #     dataset = pickle.load(f)
    # for item in dataset.keys():
    #     print(np.shape(dataset[item]))
    # for key in dataset.keys():
    #     print(np.shape(dataset[key]))

    ' -3- '
    normalise = False
    if args.eval_task_selection == 0:
        env_name_ = args.env_name
        render_ = args.render
        seed = args.seed
        output_dir_ = args.output_dir
        model_idx_ = args.model_idx
        layer_dim_ = args.layer_dim
        include_action_in_obs = args.include_action_in_obs
        num_episodes_ = args.num_episodes
        h1 = args.h1
        h2 = args.h2
        h3 = args.h3
        datapath = args.datapath
        separate_policies = False

    elif args.eval_task_selection == 1:
        ' Big step with rotation and action chunking '
        env_name_ = "VariableFriction-v7"
        output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_22_1024_2000'
        datapath = '/Users/qiyangyan/Desktop/TrainingFiles/Diffusion1/get_stacked_dataset/VFF-bigSteps-2000demos_stacked_4_2_2.pkl'
        render_ = args.render
        seed = args.seed
        include_action_in_obs = False
        separate_policies = False
        num_episodes_ = 500
        model_idx_ = 20000
        layer_dim_ = 1024
        h1 = 4
        h2 = 2
        h3 = 1

    elif args.eval_task_selection == 2:
        ' Big step with rotation and action chunking '
        env_name_ = "VariableFriction-v7"
        output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_22_1024_5000'
        datapath = '/Users/qiyangyan/Desktop/TrainingFiles/Diffusion1/get_stacked_dataset/VFF-bigSteps-5000demos_stacked_4_2_2.pkl'
        render_ = False
        seed = args.seed
        include_action_in_obs = False
        separate_policies = False
        num_episodes_ = 500
        model_idx_ = 10000
        layer_dim_ = 1024
        h1 = 4
        h2 = 2
        h3 = 2

    elif args.eval_task_selection == 3:
        """
        Description: Big step with rotation and action chunking, DR
        Training_selections: 
        """
        # TODO: Modify the robot env
        env_name_ = "VariableFriction-v7"

        # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_21_1024_1'
        # datapath = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos'

        # TODO: Use entire desired goal, uncomment the code
        # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_24_1024'
        # datapath = '/Users/qiyangyan/Desktop/TrainingFiles/Real4/Real4/demonstration/VFF-bigSteps-2000demos-randomise_1_repeat'

        # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_25_1024'
        # datapath = '/Users/qiyangyan/Desktop/TrainingFiles/Real4/Real4/demonstration/VFF-bigSteps-2000demos-randomise_1_repeat'


        render_ = args.render
        seed = args.seed
        include_action_in_obs = False
        separate_policies = False
        num_episodes_ = 500
        model_idx_ = 18700
        layer_dim_ = 1024
        h1 = 1
        h2 = 1
        h3 = 1

    elif args.eval_task_selection == 4:
        ' Big step with separate policy and manual slide termination '
        # TODO: Modify the robot env to change to demo collection is_success setting
        env_name_ = "VariableFriction-v7"

        ' Rotation + E2E 10000demos '
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Policies/models_21_1024_1',
        #                '/Users/qiyangyan/Desktop/Diffusion/Policies/models_26_1024',
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10000demos_rotation']
        # model_idx_ = [7800,
        #               1000]
        # layer_dim_ = 1024

        ' Cube cylinder: Rotation + Sliding 10000demos '
        output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_58_1024',
                       '/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_60_1024',
                       ]
        datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_cube_cylinder_slide',
                    '/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_cube_cylinder_rotation']
        model_idx_ = [3300,
                      1500]
        layer_dim_ = 1024
        normalise = True

        ' Rotation + E2E '
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Policies/models_19_1024',
        #                '/Users/qiyangyan/Desktop/Diffusion/Policies/models_26_1024'
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-1879demos-2mmPolicy-5mmThreshold',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10000demos_rotation']
        # # Slide
        # model_idx_ = [1700,
        #               1000]

        # layer_dim_ = 2048
        # normalise = True
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Policies/models_34_2048',
        #                '/Users/qiyangyan/Desktop/Diffusion/Policies/models_34_2048'
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos']
        # # Best slide + best rotate
        # model_idx_ = [400,
        #               3400]
        # Overall best
        # model_idx_ = [800,
        #               800]

        # layer_dim_ = 1024
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Policies/models_21_1024_1',
        #                '/Users/qiyangyan/Desktop/Diffusion/Policies/models_21_1024_1'
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos']
        # # # Best slide + best rotate
        # # model_idx_ = [8900,
        # #               7800]
        # # # Overall best
        # model_idx_ = [9200,
        #               9200]

        # layer_dim_ = 1024
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Policies/models_20_1024',
        #                '/Users/qiyangyan/Desktop/Diffusion/Policies/models_20_1024'
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-5000demos',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-5000demos']
        # # Best slide + best rotate
        # model_idx_ = [5400,
        #               8100]
        # # Overall best
        # # model_idx_ = [9000,
        # #               9000]

        # layer_dim_ = 1024
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Policies/models_19_1024',
        #                '/Users/qiyangyan/Desktop/Diffusion/Policies/models_19_1024'
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-1879demos-2mmPolicy-5mmThreshold',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-1879demos-2mmPolicy-5mmThreshold']
        # # Best slide + best rotate
        # # model_idx_ = [1800,
        # #               9000]
        # # Overall best
        # model_idx_ = [9800,
        #               9800]

        # layer_dim_ = 2048
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Policies/models_33_2048',
        #                '/Users/qiyangyan/Desktop/Diffusion/Policies/models_33_2048'
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos']
        # # Best slide + best rotate
        # # model_idx_ = [,
        # #               ]
        # # Overall best
        # model_idx_ = [500,
        #               500]

        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Policies/models_20_1024_1',
        #                '/Users/qiyangyan/Desktop/Diffusion/Policies/models_26_1024'
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-1879demos-2mmPolicy-5mmThreshold',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10000demos_rotation']
        # model_idx_ = [1700,
        #               1000]

        ' Rotation + slide '
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Policies/models_27_1024',
        #                '/Users/qiyangyan/Desktop/Diffusion/Policies/models_26_1024'
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/TrainingFiles/Diffusion1/demonstration/bigSteps_10000demos_slide',
        #             '/Users/qiyangyan/Desktop/TrainingFiles/Diffusion1/demonstration/bigSteps_10000demos_rotation']

        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Policies/models_32_2048',
        #                '/Users/qiyangyan/Desktop/Diffusion/Policies/models_26_1024'
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10000demos_slide',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10000demos_rotation']

        render_ = args.render
        seed = args.seed
        include_action_in_obs = False
        separate_policies = True
        num_episodes_ = args.num_episodes
        # num_episodes_ = 500
        # layer_dim_ = 2048
        h1 = 1
        h2 = 1
        h3 = 1

        'AUTO TERMINATE'
        terminate_indicator_ = args.terminate_indicator
        terminate_model_name_ = 'SimpleNN'
        termiante_save_path_ = 'models_terminate'

        'CONTROL MODE CLASSIFICATION'
        control_mode_indicator = args.control_mode_indicator

    elif args.eval_task_selection == 5:
        ' Big step with only slide policy '
        # TODO: Modify the robot env to change to demo collection is_success setting
        env_name_ = "VariableFriction-v7"
        # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_27_1024'
        output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_32_2048'
        datapath = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10000demos_slide'
        render_ = args.render
        seed = args.seed
        include_action_in_obs = False
        separate_policies = False
        num_episodes_ = 500
        model_idx_ = 1000
        layer_dim_ = 2048
        h1 = 1
        h2 = 1
        h3 = 1
        normalise = True

    elif args.eval_task_selection == 6:
        ' Big step with only slide policy '
        # TODO: Modify the robot env to change to demo collection is_success setting
        env_name_ = "VariableFriction-v7"
        output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_4_1024'
        datapath = '/Users/qiyangyan/Desktop/TrainingFiles/Diffusion1/demonstration/bigSteps_10000demos_slide'
        model_idx_ = 20000
        layer_dim_ = 1024
        render_ = args.render
        seed = args.seed
        include_action_in_obs = False
        separate_policies = False
        num_episodes_ = 500
        h1 = 1
        h2 = 1
        h3 = 1

    elif args.eval_task_selection == 7:
        ' Big Step E2E for slide'
        # TODO: Modify the robot env to change to demo collection is_success setting
        env_name_ = "VariableFriction-v7"

        # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_19_1024'
        # datapath = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-1879demos-2mmPolicy-5mmThreshold'
        # model_idx_ = 1800
        # layer_dim_ = 1024

        # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_20_1024'
        # datapath = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-5000demos'
        # model_idx_ = 5400
        # layer_dim_ = 1024

        # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_21_1024_1'
        # datapath = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos'
        # model_idx_ = 8900
        # layer_dim_ = 1024

        # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_34_2048'
        # datapath = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos'
        # model_idx_ = 400
        # layer_dim_ = 2048
        # normalise = True

        # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_33_2048'
        # datapath = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos'
        # model_idx_ = 300
        # layer_dim_ = 2048

        # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_38_1024'
        # datapath = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-bigSteps-10000demos'
        # model_idx_ = 9100
        # layer_dim_ = 1024
        # normalise = True

        output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_53_1024'
        datapath = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_10k_demos_random_slide'
        model_idx_ = 5200
        layer_dim_ = 1024
        normalise = True

        # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_46_1024'
        # datapath = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_50kdemos_slide_withJointDR'
        # model_idx_ = 2100
        # layer_dim_ = 1024
        # normalise = True

        # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_47_1024'
        # datapath = '/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_50kdemos_slide'
        # model_idx_ = 16300
        # layer_dim_ = 1024
        # normalise = True

        terminate_indicator_ = False
        control_mode_indicator = False
        terminate_model_name_ = None
        termiante_save_path_ = None
        render_ = args.render
        seed = args.seed
        include_action_in_obs = False
        separate_policies = False
        num_episodes_ = 500
        h1 = 1
        h2 = 1
        h3 = 1

    else:
        raise ValueError("Wrong Task number")

    print(" ------------------------------------------------ ")
    print("Env name: ", env_name_)
    print("Model: ", output_dir_)
    print("Data: ", datapath)
    print("Separate: ", separate_policies)
    print("Model_idx: ", model_idx_)
    print("Seed: ", seed)
    print("Norm: ", normalise)
    print("Auto Termiante: ", terminate_indicator_)
    print("Control Mode: ", control_mode_indicator)
    print(" ------------------------------------------------ ")
    input("Press to Continue")

    if not separate_policies:
        with open(datapath, 'rb') as f:
            dataset = pickle.load(f)

        if len(dataset['desired_goals'][0]) == 9:
            pass
        else:
            assert len(dataset['desired_goals'][0]) == 11
            print("Desired goal = (11,0)")
            dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])

        for item in dataset.keys():
            print(np.shape(dataset[item]))
        for key in dataset.keys():
            print(np.shape(dataset[key]))

        ' Measure the number of demos '
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print(num)

    else:
        dataset = {
            "dict1": {},
            "dict2": {}
        }
        for i, key in enumerate(dataset):
            print(key)
            # Load data from pickle file
            with open(datapath[i], 'rb') as f:
                dataset[key] = pickle.load(f)

            # Check the length of 'desired_goals' in the loaded dictionary
            if len(dataset[key]['desired_goals'][0]) == 9:
                pass
            elif len(dataset[key]['desired_goals'][0]) == 11:
                print("Desired goal = (11,0)")
                # Correct the shape of 'desired_goals'
                dataset[key]['desired_goals'] = np.array(
                    [sub_array[:-2] for sub_array in dataset[key]['desired_goals']])
            else:
                raise ValueError("Unexpected length of 'desired_goals'")  # Handling unexpected cases

        ' Measure the number of demos '
        num = 0
        for key in dataset:
            for i, item in enumerate(dataset[key]['terminals']):
                if item:
                    num += 1
            print(num)

    # TODO: Remember to modify the self.r_threshold=0.002 back to 0.005
    # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_17'
    # output_dir_ = '/Users/qiyangyan/Desktop/Diffusion/Policies/models_15_1024'
    # output_dir = "./results/"
    # model_idx_ = 7000
    # layer_dim_ = 1024
    # with open(f'/Users/qiyangyan/Desktop/Diffusion/Demonstration/VFF-2mmPolicy_2mmThreshold', 'rb') as f:
    #     dataset = pickle.load(f)
    # for item in dataset.keys():`
    #     print(np.shape(dataset[item]))

    print("Separate policy: ", separate_policies)

    if separate_policies:
        data_4_norm = datapath[0]
    else:
        data_4_norm = datapath
    evaluation = EvaluateDiffusion(env_name=env_name_,
                                   render=render_,
                                   dataset=dataset,
                                   output_dir=output_dir_,
                                   model_idx=model_idx_,
                                   seed=seed,
                                   layer_dim=layer_dim_,
                                   include_action_in_obs=include_action_in_obs,
                                   separate_policies=separate_policies,
                                   terminate_indicator=terminate_indicator_,
                                   terminate_model_name=terminate_model_name_,
                                   termiante_save_path=termiante_save_path_,
                                   control_mode_indicator=control_mode_indicator,
                                   )
    evaluation.eval_diffusion_with_rotation(num_episodes=num_episodes_,
                                            num_steps=11,
                                            demonstration_file_name='trajectory',
                                            store_trajectory=False,
                                            h1=h1,
                                            h2=h2,
                                            h3=h3,
                                            normalise=normalise,
                                            data_4_norm=data_4_norm
                                            )

    print(" ------------------------------------------------ ")
    print("Model: ", output_dir_)
    print("Data: ", datapath)
    print("Separate: ", separate_policies)
    print("Model_idx: ", model_idx_)
    print("Seed: ", seed)
    print("Auto Termiante: ", terminate_indicator_)
    print(" ------------------------------------------------ ")
