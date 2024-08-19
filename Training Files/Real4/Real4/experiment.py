from mpi4py import MPI
import argparse
import pickle
import numpy as np
import os
from evaluate_diffusion import EvaluateDiffusion

def run_evaluation_for_seed(seed_idx, args):
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
    # (Continue setting up other configurations as per the provided code...)

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

        ' Cube cylinder: Rotation + Sliding 10000 demos '
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_58_1024',
        #                '/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_60_1024',
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_cube_cylinder_slide',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_cube_cylinder_rotation']
        # model_idx_ = [3300,
        #               1500]
        # layer_dim_ = 1024
        # normalise = True

        ' Cube cylinder: Rotation + Sliding 10000 demos + No Obj Vel'
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_62_1024',
        #                '/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_63_1024',
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_cube_cylinder_slide_noObjVel',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_cube_cylinder_rotation_noObjVel']
        # model_idx_ = [3300,
        #               1500]
        # layer_dim_ = 1024
        # normalise = True
        # object = "cube_cylinder"

        ' Three cylinder : Rotation + Sliding 10000 demos '
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_64_1024',
        #                '/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_65_1024',
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_three_cylinder_slide_noObjVel',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demo_random/train_bigSteps_10k_three_cylinder_rotation_noObjVel']
        # model_idx_ = [3200,
        #               300]
        # layer_dim_ = 1024
        # normalise = True
        # object = "three_cylinder"

        ' Cube 1cm : Rotation + Sliding 10000 demos '
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_66_1024',
        #                '/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_67_1024',
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube1cm/train_10k_cube_1cm_noObjVel_slide',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube1cm/train_10k_cube_1cm_noObjVel_rotation']
        # model_idx_ = [2600,
        #               500]
        # layer_dim_ = 1024
        # normalise = True
        # object = "cube_1cm"

        ' Cube 2cm : Rotation + Sliding 10000 demos '
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_68_1024',
        #                '/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_69_1024',
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube2cm/train_10k_cube_2cm_noObjVel_slide',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube2cm/train_10k_cube_2cm_noObjVel_rotation']
        # model_idx_ = [1400,
        #               2500]
        # layer_dim_ = 1024
        # normalise = True
        # object = "cube_2cm"

        ' Mixed object: Rotation + Sliding 10000 demos each shape '
        # output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random_mix/models_70_1024',
        #                '/Users/qiyangyan/Desktop/Diffusion/Demo_random_mix/models_71_1024',
        #                ]
        # datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/mixed_object/train_40k_noObjVel_slide',
        #             '/Users/qiyangyan/Desktop/Diffusion/Demo_random/mixed_object/train_40k_noObjVel_rotation']
        # model_idx_ = [5800,
        #               100]
        # layer_dim_ = 1024
        # normalise = True
        # object = "mixed_object"

        ' Co-train: Rotation + Sliding 10000 demos each shape '
        output_dir_ = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random_mix/models_74_1024',
                       '/Users/qiyangyan/Desktop/Diffusion/Demo_random/models_67_1024',
                       ]
        datapath = ['/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube1cm/train_10k_cube_1cm_noObjVel_slide',
                    '/Users/qiyangyan/Desktop/Diffusion/Demo_random/cube1cm/train_10k_cube_1cm_noObjVel_rotation']
        model_idx_ = [9500,
                      500]
        layer_dim_ = 1024
        normalise = True
        object = "cube_1cm"

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

    if not separate_policies:
        with open(datapath, 'rb') as f:
            dataset = pickle.load(f)
        if len(dataset['desired_goals'][0]) == 9:
            pass
        else:
            assert len(dataset['desired_goals'][0]) == 11
            dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])
        for item in dataset.keys():
            print(np.shape(dataset[item]))
    else:
        dataset = {
            "dict1": {},
            "dict2": {}
        }
        for i, key in enumerate(dataset):
            with open(datapath[i], 'rb') as f:
                dataset[key] = pickle.load(f)
            if len(dataset[key]['desired_goals'][0]) == 9:
                pass
            elif len(dataset[key]['desired_goals'][0]) == 11:
                dataset[key]['desired_goals'] = np.array(
                    [sub_array[:-2] for sub_array in dataset[key]['desired_goals']])
            else:
                raise ValueError("Unexpected length of 'desired_goals'")
        for key in dataset:
            for i, item in enumerate(dataset[key]['terminals']):
                if item:
                    num += 1
            print(num)

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
                                   seed_idx=seed_idx,
                                   real=args.real)

    eval_history = evaluation.eval_diffusion_with_rotation(num_episodes=num_episodes_,
                                                           num_steps=11,
                                                           demonstration_file_name='trajectory',
                                                           store_trajectory=False,
                                                           h1=h1,
                                                           h2=h2,
                                                           h3=h3,
                                                           normalise=normalise,
                                                           data_4_norm=data_4_norm,
                                                           object=object,
                                                           real=args.real)
    return eval_history


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
    parser.add_argument('--seed_idx', type=int, default=0, help='Seed value')
    parser.add_argument('--random_seed_list', action='store_true', help='Use random seed list for paper experiment')
    parser.add_argument('--real', action='store_true', help='Test on real robot')
    args = parser.parse_args()

    seed_idx_list = []
    if args.random_seed_list is True:
        seed_idx_list = [70, 123, 354]
    else:
        seed_idx_list.append(args.seed_idx)

    eval_history_dict = {
        "pos_error_list": [],
        "orientation_error_list": [],
        "pos_error_list_include_failure": [],
        "orientation_error_list_include_failure": [],
        "success_rate": [],
        "success": [],
    }

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        seeds_to_process = seed_idx_list
    else:
        seeds_to_process = None

    seeds_to_process = comm.scatter(seeds_to_process, root=0)

    eval_history = run_evaluation_for_seed(seeds_to_process, args)

    gathered_histories = comm.gather(eval_history, root=0)

    if rank == 0:
        for history in gathered_histories:
            for key in history.keys():
                eval_history_dict[key].append(history[key])

        statistics = {key: {} for key in eval_history_dict}
        for key, values in eval_history_dict.items():
            all_values = np.concatenate(values)
            statistics[key]['mean'] = np.mean(all_values)
            statistics[key]['std'] = np.std(all_values)
            statistics[key]['median'] = np.median(all_values)

        print(statistics)
