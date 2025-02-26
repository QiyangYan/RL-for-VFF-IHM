import pickle  
import argparse  
import torch
import os
from utils.data_sampler import Data_Sampler
from utils import utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def normalize(data, mean, std):
    # Ensure standard deviation is not zero to avoid division by zero error
    std = np.where(std == 0, 1, std)
    return (data - mean) / std

def normalize_max_min(data, min, max):
    # Ensure standard deviation is not zero to avoid division by zero error
    mask = (max == min)
    # data_norm = np.where(mask, 0.5, (data - min) / (max - min))
    data_norm = np.divide(data - min, max - min, where=~mask, out=np.full(data.shape, np.nan))
    data_norm = np.where(mask, 0.5, data_norm)
    return data_norm * 2 - 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--max_action", default=1., type=float)
    parser.add_argument("--num_epochs", default=10000, type=int)
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--training_selection", default=1, type=int)
    parser.add_argument("--layer_dim", default=256, type=int)
    parser.add_argument("--num_demos", default=0, type=int)
    parser.add_argument("--h1", default=4, type=int)
    parser.add_argument("--h2", default=2, type=int)
    parser.add_argument("--h3", default=2, type=int)
    parser.add_argument("--cotrain_ratio", default=0.5, type=float)
    parser.add_argument("--add_noise", action='store_true', help='Add noise to observation')
    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(args.device)

    cotrain = False

    if args.training_selection == 1:
        # Same observation space 
        print("Same observation space")
        with open(f'dataset/VFF-1686demos', 'rb') as f:
            dataset = pickle.load(f)
        dataset['desired_goals'] = dataset['desired_goals_radi']
        for key in dataset.keys():
            print(np.shape(dataset[key]))
        
    elif args.training_selection == 2:
        # Pose desired goal
        print("Pose desired goal")
        with open(f'dataset/VFF-1686demos', 'rb') as f:
            dataset = pickle.load(f)
        for key in dataset.keys():
            print(np.shape(dataset[key]))

    elif args.training_selection == 3:
        # Pose desired goal with norm
        print("Normalize")
        with open(f'dataset/VFF-1686demos', 'rb') as f:
            dataset = pickle.load(f)
        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        for key in dataset.keys():
            print(f"{key}: ", np.shape(dataset[key]))
        
    elif args.training_selection == 4:
        # More informative observation space
        print("Informative observation space")
        with open(f'dataset/VFF-1817demos', 'rb') as f:
            dataset = pickle.load(f)
        for key in dataset.keys():
            print(np.shape(dataset[key]))

    elif args.training_selection == 5:
        # Normalized observation space
        with open(f'dataset/VFF-1817demos', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        print(goal_mean, goal_std)
        
        for key in dataset.keys():
            print(np.shape(dataset[key]))

    elif args.training_selection == 6:
        # Add Rotation
        print("Add Rotation 5mm threshold")
        with open(f'dataset/VFF-2mmPolicy_5mmThreshold', 'rb') as f:
            dataset = pickle.load(f)
        for key in dataset.keys():
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 7:
        # Add Rotation
        print("Add Rotation 2mm threshold")
        with open(f'dataset/VFF-2mmPolicy_2mmThreshold', 'rb') as f:
            dataset = pickle.load(f)
        for key in dataset.keys():
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 8:
        # 5947 demos without rotation
        print("Informative observation space")
        with open(f'dataset/VFF-5947demos', 'rb') as f:
            dataset = pickle.load(f)
        for key in dataset.keys():
            print(np.shape(dataset[key]))
    
    elif args.training_selection == 9:
        # 5947 demos without rotation with normalization
        print("Informative observation space")
        with open(f'dataset/VFF-5947demos', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        print(goal_mean, goal_std)
        
        for key in dataset.keys():
            print(np.shape(dataset[key]))

    elif args.training_selection == 10:
        # 10000 demos with rotation
        print("Add Rotation 2mm threshold - 10000 demos")
        with open(f'dataset/VFF-2mmPolicy_2mmThreshold', 'rb') as f:
            dataset = pickle.load(f)
        for key in dataset.keys():
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
    
    elif args.training_selection == 11:
        # 10000 demos with rotation - with actor-1200 as pretrained policy from last 10000 demos
        print("Add Rotation 2mm threshold - 10000 demos with pretrain")
        with open(f'dataset/VFF-2mmPolicy_2mmThreshold', 'rb') as f:
            dataset = pickle.load(f)
        for key in dataset.keys():
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 12:
        print(f"Add Rotation 2mm threshold - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-2mmPolicy_2mmThreshold', 'rb') as f:
            dataset = pickle.load(f)
        for key in dataset.keys():
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 13:
        print(f"Only predict continuous term - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-2mmPolicy_2mmThreshold', 'rb') as f:
            dataset = pickle.load(f)
        for key in dataset.keys():
            print(np.shape(dataset[key]))
        continue_action = dataset['actions'][:,0].reshape(-1, 1)
        dataset['actions'] = continue_action
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
    
    elif args.training_selection == 14:
        # Random policy without rotation
        print(f"Random Policy without Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-random_withoutRotation_9983demos', 'rb') as f:
            dataset = pickle.load(f)
        for key in dataset.keys():
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 15:
        # Random policy without rotation
        # obs + action
        print(f"Add last_action into Obs - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-random_withoutRotation_9983demos', 'rb') as f:
            dataset = pickle.load(f)

        first_action = np.zeros((1, dataset['actions'].shape[1]))
        all_actions = np.vstack((first_action, dataset['actions'][:-1]))
        obs_with_action = np.hstack((dataset['observations'], all_actions))

        # Replace first action for every episode
        episode_ends = [index for index, value in enumerate(dataset['terminals']) if value]
        for idx in episode_ends[:-1]:
            obs_with_action[idx+1][-2:] =  np.zeros((1, dataset['actions'].shape[1]))
        dataset['observations'] = obs_with_action

        for key in dataset.keys():
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 16:
        # Random policy with rotation
        print(f"Random Policy with Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-random_9884demos', 'rb') as f:
            dataset = pickle.load(f)
        for key in dataset.keys():
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 17:
        # Random policy with rotation and action in obs
        # obs + action
        print(f"Add last_action into Obs with Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-random_9884demos', 'rb') as f:
            dataset = pickle.load(f)

        first_action = np.zeros((1, dataset['actions'].shape[1]))
        all_actions = np.vstack((first_action, dataset['actions'][:-1]))
        obs_with_action = np.hstack((dataset['observations'], all_actions))

        # Replace first action for every episode
        episode_ends = [index for index, value in enumerate(dataset['terminals']) if value]
        for idx in episode_ends[:-1]:
            obs_with_action[idx+1][-2:] =  np.zeros((1, dataset['actions'].shape[1]))
        dataset['observations'] = obs_with_action

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
    
    elif args.training_selection == 18:
        # Random policy with rotation
        print(f"Add last_action into Obs with Rotation RL - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-2mmPolicy_2mmThreshold', 'rb') as f:
            dataset = pickle.load(f)

        first_action = np.zeros((1, dataset['actions'].shape[1]))
        all_actions = np.vstack((first_action, dataset['actions'][:-1]))
        obs_with_action = np.hstack((dataset['observations'], all_actions))

        # Replace first action for every episode
        episode_ends = [index for index, value in enumerate(dataset['terminals']) if value]
        for idx in episode_ends[:-1]:
            obs_with_action[idx+1][-2:] =  np.zeros((1, dataset['actions'].shape[1]))
        dataset['observations'] = obs_with_action

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
    
    elif args.training_selection == 19:
        # Big Step with rotation
        print(f"Big step with rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-bigSteps-1879demos-2mmPolicy-5mmThreshold', 'rb') as f:
            dataset = pickle.load(f)

        # remove the last two term from each desired goal
        dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 20:
        # Big Step with rotation 5000 demos
        print(f"Big step with rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-bigSteps-5000demos', 'rb') as f:
            dataset = pickle.load(f)

        # remove the last two term from each desired goal
        dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 21:
        # Big Step with rotation 10000 demos
        print(f"Big step with rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-bigSteps-10000demos', 'rb') as f:
            dataset = pickle.load(f)

        # remove the last two term from each desired goal
        dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 22:
        # Big Step with rotation args.num_demos demos, last two goals is removed during the dataset preparation
        print(f"Big step with rotation and action chunking - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-bigSteps-{args.num_demos}demos_stacked_4_2_2.pkl', 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: /VFF-bigSteps_{args.num_demos}demos_stacked_{args.h1}_{args.h2}_{args.h3}")

    elif args.training_selection == 23:
        # Big Step with rotation args.num_demos demos, last two goals is removed during the dataset preparation
        print(f"Big step with rotation and action chunking - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-bigSteps-{args.num_demos}demos_stacked_{args.h1}_{args.h2}_{args.h3}.pkl', 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: /VFF-bigSteps_{args.num_demos}demos_stacked_{args.h1}_{args.h2}_{args.h3}")

    elif args.training_selection == 24:
        # Demo with DR
        print(f"Big step with rotation and DR - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-bigSteps-2000demos-randomise_1_repeat', 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: VFF-bigSteps-2000demos-randomise_1_repeat with DR with actual achieved goal")
    
    elif args.training_selection == 25:
        # Demo with DR
        print(f"Big step with rotation and DR - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-bigSteps-2000demos-randomise_1_repeat', 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['sampled_desired_goals']])
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: VFF-bigSteps-2000demos-randomise_1_repeat with DR with sampled desired goal")
    
    elif args.training_selection == 26:
        # Isolate rotation part as an individual policy
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_rotation', 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_rotation")

    elif args.training_selection == 27:
        # Isolate sliding part as an individual policy
        # Modify out_dir_ if needed
        print(f"Big step only Sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide', 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide")

    elif args.training_selection == 28:
        # Sliding with end indicator
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide_endIndicator', 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide_endIndicator")

    elif args.training_selection == 29:
        # Sliding
        # + Bigger network
        print(f"Big step only Sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide', 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide")
    
    elif args.training_selection == 30:
        # Rotation 
        # + Bigger network
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_rotation', 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_rotation")

    elif args.training_selection == 31:
        # Sliding + Indicator
        # + Bigger network
        print(f"Big step only Sliding and indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide_endIndicator', 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide_endIndicator")

    elif args.training_selection == 32:
        # Sliding
        # + Bigger network
        # + Norm
        print(f"Big step only Sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide")
    
    elif args.training_selection == 33:
        # E2E
        # + Bigger network
        print(f"Big step with rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-bigSteps-10000demos', 'rb') as f:
            dataset = pickle.load(f)

        # remove the last two term from each desired goal
        dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos")
    
    elif args.training_selection == 34:
        # E2E
        # + Bigger network
        # + Norm 
        print(f"Big step with rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-bigSteps-10000demos', 'rb') as f:
            dataset = pickle.load(f)

        # remove the last two term from each desired goal
        dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos")

    elif args.training_selection == 35:
        # Sliding with end indicator
        # + Bigger network
        # + Norm
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide_endIndicator', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide_endIndicator")

    elif args.training_selection == 36:
        # E2E
        # + correct rotation action mapping
        # + Bigger network: 2048
        # + Norm 
        print(f"Big step with rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/VFF-bigSteps-10000demos', 'rb') as f:
            dataset = pickle.load(f)

        # remove the last two term from each desired goal
        dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])
        # map the rotation to correct range
        for i, terminal in enumerate(dataset['terminals']):
            if terminal == 1:
                dataset['actions'][i][1] = dataset['actions'][i][1] * 2 - 1

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: VFF-bigSteps-10000demos")
    
    elif args.training_selection == 37:
        # Rotation
        # + correct rotation action mapping
        # + network: 1024
        # + Norm 
        print(f"Big step with rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_rotation', 'rb') as f:
            dataset = pickle.load(f)

        # remove the last two term from each desired goal
        dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])
        # map the rotation to correct range
        for i, terminal in enumerate(dataset['terminals']):
            if terminal == 1:
                dataset['actions'][i][1] = dataset['actions'][i][1] * 2 - 1

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_rotation")

    elif args.training_selection == 38:
        # Sliding
        # + network: 1024
        # + Norm 
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide")

    elif args.training_selection == 39:
        # Sliding
        # + action norm
        # + network: 1024
        # + Norm 
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action_mean = np.mean(dataset['actions'], axis=0)
        action_std = np.std(dataset['actions'], axis=0)
        action_norm = normalize(dataset['actions'], action_mean, action_std)
        dataset['actions'] = action_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide")

    elif args.training_selection == 40:
        # NOOOO
        # Sliding with end indicator
        # + action norm
        # + network: 1024
        # + Norm

        '''
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide_endIndicator', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action_mean = np.mean(dataset['actions'], axis=0)
        action_std = np.std(dataset['actions'], axis=0)
        action_norm = normalize(dataset['actions'], action_mean, action_std)
        dataset['actions'] = action_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide_endIndicator")
        '''
        raise ValueError("Invalid training selection")

    elif args.training_selection == 41:
        # Sliding
        # + action norm (max, min)
        # + network: 1024
        # + Norm (max, min)
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_max = np.max(dataset['desired_goals'], axis=0)
        goal_min = np.min(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize_max_min(dataset['desired_goals'], min=goal_min, max=goal_max)
        dataset['desired_goals'] = desired_goal_norm

        obs_max = np.max(dataset['observations'], axis=0)
        obs_min = np.min(dataset['observations'], axis=0)
        obs_norm = normalize_max_min(dataset['observations'], min=obs_min, max=obs_max)
        dataset['observations'] = obs_norm

        action_max = np.max(dataset['actions'], axis=0)
        action_min = np.min(dataset['actions'], axis=0)
        action_norm = normalize_max_min(dataset['actions'], min=action_min, max=action_max)
        dataset['actions'] = action_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide")

    elif args.training_selection == 42:
        # Sliding
        # + network: 1024
        # + Norm (max, min)
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_max = np.max(dataset['desired_goals'], axis=0)
        goal_min = np.min(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize_max_min(dataset['desired_goals'], min=goal_min, max=goal_max)
        dataset['desired_goals'] = desired_goal_norm

        obs_max = np.max(dataset['observations'], axis=0)
        obs_min = np.min(dataset['observations'], axis=0)
        obs_norm = normalize_max_min(dataset['observations'], min=obs_min, max=obs_max)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide")

    elif args.training_selection == 43:
        # Sliding with end indicator
        # + action norm (max, min)
        # + network: 1024
        # + Norm (max, min)
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide_endIndicator', 'rb') as f:
            dataset = pickle.load(f)

        goal_max = np.max(dataset['desired_goals'], axis=0)
        goal_min = np.min(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize_max_min(dataset['desired_goals'], min=goal_min, max=goal_max)
        dataset['desired_goals'] = desired_goal_norm

        obs_max = np.max(dataset['observations'], axis=0)
        obs_min = np.min(dataset['observations'], axis=0)
        obs_norm = normalize_max_min(dataset['observations'], min=obs_min, max=obs_max)
        dataset['observations'] = obs_norm

        action_max = np.max(dataset['actions'], axis=0)
        action_min = np.min(dataset['actions'], axis=0)
        action_norm = normalize_max_min(dataset['actions'], min=action_min, max=action_max)
        dataset['actions'] = action_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide_endIndicator")
    
    elif args.training_selection == 44:
        # Sliding with end indicator
        # + network: 1024
        # + Norm (max, min)
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide_endIndicator', 'rb') as f:
            dataset = pickle.load(f)

        goal_max = np.max(dataset['desired_goals'], axis=0)
        goal_min = np.min(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize_max_min(dataset['desired_goals'], min=goal_min, max=goal_max)
        dataset['desired_goals'] = desired_goal_norm

        obs_max = np.max(dataset['observations'], axis=0)
        obs_min = np.min(dataset['observations'], axis=0)
        obs_norm = normalize_max_min(dataset['observations'], min=obs_min, max=obs_max)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide_endIndicator")

    elif args.training_selection == 46:
        # Sliding
        # + network: 1024
        # + Norm
        # 50k demo
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_50kdemos_slide_withJointDR', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: bigSteps_50kdemos_slide_withJointDR")
    
    elif args.training_selection == 47:
        # Sliding
        # + network: 1024
        # + Norm
        # 50k demo
        # No DR
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_50kdemos_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: bigSteps_50kdemos_slide")

    elif args.training_selection == 48:
        # Sliding
        # + network: 1024
        # + Norm
        # 50k demo
        # No DR
        # Isolated End Indicator
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_50kdemos_slide_endIndicator', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        continue_action = dataset['actions'][:,2].reshape(-1, 1)
        dataset['actions'] = continue_action

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_50kdemos_slide_endIndicator")

    elif args.training_selection == 49:
        # Sliding
        # + network: 1024
        # + Norm
        # 50k demo
        # No DR
        # Only continuous
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_50kdemos_slide_endIndicator', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        continue_action = dataset['actions'][:,0].reshape(-1, 1)
        dataset['actions'] = continue_action

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_50kdemos_slide_endIndicator")

    elif args.training_selection == 50:
        # Sliding
        # + network: 1024
        # + Norm
        # 50k demo
        # No DR
        # Only discrete control mode
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_50kdemos_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        continue_action = dataset['actions'][:,1].reshape(-1, 1)
        dataset['actions'] = continue_action

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_50kdemos_slide")

    elif args.training_selection == 51:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + only terminate
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide_endIndicator', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        continue_action = dataset['actions'][:,2].reshape(-1, 1)
        dataset['actions'] = continue_action

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_slide_endIndicator")

    elif args.training_selection == 52:
        # Sliding
        # + network: 1024
        # + Norm
        # + 50k demo
        # + No DR
        # + Isolated End Indicator
        # + 20000
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_50kdemos_slide_endIndicator', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        continue_action = dataset['actions'][:,2].reshape(-1, 1)
        dataset['actions'] = continue_action

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_50kdemos_slide_endIndicator")
    
    elif args.training_selection == 53:
        # Sliding
        # + random demo collection
        # + network: 1024
        # + Norm
        # + 10k demo
        # + No DR
        print(f"Big step with slide - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10k_demos_random_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)

    elif args.training_selection == 54:
        # Sliding
        # + random demo collection
        # + network: 1024
        # + Norm
        # + 10k demo
        # + End
        # + 30000 epoch
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10k_demos_random_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        dataset['actions'] = np.expand_dims(dataset['terminals'].copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("bigSteps_10k_demos_random_slide")
        print("End indicator")

    elif args.training_selection == 55:
        print("Predict the start pose for rotation")
        dir = 'dataset/rotation_start_pose_prediction_dataset'
        with open(dir, 'rb') as f:
            dataset = pickle.load(f)

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        print("Number of demos: ", args.num_demos)
        print("rotation_start_pose_prediction")
        print(dir)

    elif args.training_selection == 56:
        # Sliding
        # + random demo collection
        # + network: 1024
        # + Norm
        # + 10k demo
        # + End
        # + 30000 epoch
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10k_demos_random_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("bigSteps_10k_demos_random_slide")
        print("End indicator")

    elif args.training_selection == 57:
        # Sliding
        # + random demo collection
        # + network: 1024
        # + Norm
        # + 10k demo
        # + End
        # + 30000 epoch
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'dataset/train.pkl', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("train dataset")

    elif args.training_selection == 58:
        # Sliding
        # + network: 1024
        # + Norm
        # 10k demo
        # + cube cylinder
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/train_bigSteps_10k_cube_cylinder_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: cube cylinder only sliding")
    
    elif args.training_selection == 59:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: cube only sliding")

    elif args.training_selection == 60:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube cylinder
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/train_bigSteps_10k_cube_cylinder_rotation', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_bigSteps_10k_cube_cylinder_rotation")

    elif args.training_selection == 61:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/bigSteps_10000demos_rotation', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: bigSteps_10000demos_rotation")

    elif args.training_selection == 62:
        # Sliding
        # + network: 1024
        # + Norm
        # 10k demo
        # + cube cylinder
        # + remove obs velocity in obs
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/train_bigSteps_10k_cube_cylinder_slide_noObjVel', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: cube cylinder only sliding without object velocity")
        print("train_bigSteps_10k_cube_cylinder_slide_noObjVel")

    elif args.training_selection == 63:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube cylinder
        # + remove obs velocity in obs
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/train_bigSteps_10k_cube_cylinder_rotation_noObjVel', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_bigSteps_10k_cube_cylinder_rotation_noObjVel")

    elif args.training_selection == 64:
        # Sliding
        # + network: 1024
        # + Norm
        # 10k demo
        # + three cylinder
        # remove obs velocity in obs
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/train_bigSteps_10k_three_cylinder_slide_noObjVel', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: three cylinder only sliding without object velocity")
        print("train_bigSteps_10k_three_cylinder_slide_noObjVel")

    elif args.training_selection == 65:
        # Rotation
        # + network: 1024
        # + Norm
        # + 10k demo
        # + three cylinder
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/train_bigSteps_10k_three_cylinder_rotation_noObjVel', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_bigSteps_10k_three_cylinder_rotation_noObjVel")

    elif args.training_selection == 66:
        # Sliding
        # + network: 1024
        # + Norm
        # 10k demo
        # + cube1cm
        # remove obs velocity in obs
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube1cm/train_10k_cube_1cm_noObjVel_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_10k_cube_1cm_noObjVel_slide")

    elif args.training_selection == 67:
        # Rotation
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube1cm
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube1cm/train_10k_cube_1cm_noObjVel_rotation', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_10k_cube_1cm_noObjVel_rotation")

    elif args.training_selection == 68:
        # Sliding
        # + network: 1024
        # + Norm
        # 10k demo
        # + cube2cm
        # remove obs velocity in obs
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube2cm/train_10k_cube_2cm_noObjVel_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_10k_cube_2cm_noObjVel_slide")

    elif args.training_selection == 69:
        # Rotation
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube2cm
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube2cm/train_10k_cube_2cm_noObjVel_rotation', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_10k_cube_2cm_noObjVel_rotation")

    elif args.training_selection == 70:
        # Sliding
        # + network: 1024
        # + Norm
        # 10k demo
        # + cube2cm
        # remove obs velocity in obs
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/mixed_object/train_40k_noObjVel_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_40k_noObjVel_slide")

    elif args.training_selection == 71:
        # Rotation
        # + network: 1024
        # + Norm
        # + 40k demo
        # + mixed object
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/mixed_object/train_40k_noObjVel_rotation', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_40k_noObjVel_rotation")

    elif args.training_selection == 72:
        # with velocity
        print(f"Big step with rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube/train_10k_slide', 'rb') as f:
            dataset = pickle.load(f)

        # remove the last two term from each desired goal
        dataset['desired_goals'] = np.array([sub_array[:-2] for sub_array in dataset['desired_goals']])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 73:
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube/train_10k_slide_for_real', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 74:
        # Sliding
        # + network: 1024
        # + Norm
        # 10k demo
        # + cube1cm
        # remove obs velocity in obs
        # cotrain
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube1cm/train_10k_cube_1cm_noObjVel_slide', 'rb') as f:
            dataset = pickle.load(f)

        with open(f'dataset/mixed_object/train_30k_noObjVel_slide.pkl', 'rb') as f:
            dataset_cotrain = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        action_cotrain = dataset_cotrain['actions'][:,0]
        dataset_cotrain['actions'] = np.expand_dims(action_cotrain.copy(), axis=1)

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        for key in dataset_cotrain.keys(): 
            print(np.shape(dataset_cotrain[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_10k_cube_1cm_noObjVel_slide")
    
    elif args.training_selection == 75:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_cylinder/train_10k_cube_cylinder_slide_for_real', 'rb') as f:
            dataset = pickle.load(f)
        
        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        cotrain = False

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 76:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_cylinder/train_10k_cube_cylinder_slide_for_real', 'rb') as f:
            dataset = pickle.load(f)
        
        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        cotrain = False

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 77:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube/bigSteps_10000demos_slide_for_real', 'rb') as f:
            dataset = pickle.load(f)
        
        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        cotrain = False

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 78:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        with open(f'dataset/real_cube/real_cube_1h_train_train', 'rb') as f:
            dataset = pickle.load(f)
        
        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        cotrain = False

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 79:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        with open(f'dataset/real_cube/real_cube_1h_train_train', 'rb') as f:
            dataset = pickle.load(f)
        
        with open(f'dataset/cube/bigSteps_10000demos_slide_for_real_final', 'rb') as f:
            dataset_cotrain = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        action_cotrain = dataset_cotrain['actions'][:,0]
        dataset_cotrain['actions'] = np.expand_dims(action_cotrain.copy(), axis=1)

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)

    elif args.training_selection == 80:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        with open(f'dataset/real_cube/real_cube_1h_train_slide', 'rb') as f:
            dataset = pickle.load(f)
        
        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        cotrain = False

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 81:
        print(f"rotation for real - network layer dimension: {args.layer_dim}")
        with open(f'dataset/real_cube/real_cube_1h_train_rotation', 'rb') as f:
            dataset = pickle.load(f)
        
        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        cotrain = False

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 82:
        print(f"rotation for real - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube/bigSteps_10000demos_rotation_for_real_final', 'rb') as f:
            dataset = pickle.load(f)
        
        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        cotrain = False

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 83:
        
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube/bigSteps_100demos_slide_for_real_final', 'rb') as f:
            dataset = pickle.load(f)
        
        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        cotrain = False

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)

    elif args.training_selection == 84:
        # cube
        # slide
        # obs = 15
        # goal = 7
        
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube/train_10k_cube_slide_for_real', 'rb') as f:
            dataset = pickle.load(f)
        
        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        cotrain = False

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("TASK: train_10k_cube_slide_for_real")

    elif args.training_selection == 85:
        # cube
        # rotation
        # obs = 15
        # goal = 7
        
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube/train_10k_cube_rotation_for_real', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print(f"TASK: train_10k_cube_rotation_for_real")

    elif args.training_selection == 86:
        # Sliding
        # + network: 1024
        # + Norm
        # 10k demo
        # + cube cylinder
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: cube cylinder only sliding without object velocity")
        print("train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal")
        
    elif args.training_selection == 87:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube cylinder
        # + remove obs velocity in obs
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_rotation_noObjVel_forReal', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_bigSteps_10k_cube_cylinder_rotation_noObjVel_forReal")
        
    elif args.training_selection == 88:
        # Sliding
        # + network: 1024
        # + Norm
        # 10k demo
        # + cube cylinder
        # + remove obs velocity in obs
        # + for real
        # + remove small actuators
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal_noSmall', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: cube cylinder only sliding without object velocity + no small actuator")
        print("train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal_noSmall")

    elif args.training_selection == 89:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube cylinder
        # + remove obs velocity in obs
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_rotation_noObjVel_forReal_noSmall', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_bigSteps_10k_cube_cylinder_rotation_noObjVel_forReal_noSmall")

    elif args.training_selection == 90:
        # Sliding
        # + network: 1024
        # + Norm
        # 10k demo
        # + cube cylinder
        # + remove obs velocity in obs
        # + for real
        # + add noise
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: cube cylinder only sliding without object velocity")
        print("train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal")

    elif args.training_selection == 91:
        # Sliding
        # + network: 1024
        # + Norm
        # 10k demo
        # + three cylinder
        # + remove obs velocity in obs
        # + for real
        # + add noise
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/three_cylinder/train_bigSteps_10k_three_cylinder_slide_noObjVel_forReal_noSmall', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: three cylinder only sliding without object velocity")
        print("train_bigSteps_10k_three_cylinder_slide_noObjVel_forReal_noSmall")

    elif args.training_selection == 92:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + three cylinder
        # + remove obs velocity in obs
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/three_cylinder/train_bigSteps_10k_three_cylinder_rotation_noObjVel_forReal_noSmall', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_bigSteps_10k_three_cylinder_rotation_noObjVel_forReal_noSmall")

    elif args.training_selection == 93:
        # Isolate sliding part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube cylinder
        # + remove obs velocity in obs
        # + no small actuataor info
        # + no radi and orientation info
        # + noise
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal_noSmall', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm[:, :7]

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm[:, :-3]

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: three cylinder only sliding without object velocity and no small")
        print("train_bigSteps_10k_three_cylinder_slide_noObjVel_forReal_noSmall")

        # print("obs_std: ", obs_std)

    elif args.training_selection == 94:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube cylinder
        # + remove obs velocity in obs
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_rotation_noObjVel_forReal_noSmall', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm[:, :7]

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm[:, :-3]

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_bigSteps_10k_cube_cylinder_rotation_noObjVel_forReal_noSmall")


    elif args.training_selection == 95:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        # data_path = 'dataset/real demo/cube_cylinder/train_100_cube_cylinder_real_slide'
        # with open(data_path, 'rb') as f:
        #     dataset = pickle.load(f)
        data_path = 'dataset/real demo/cube_cylinder_200/train_200_cube_cylinder_real_slide_work'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        # data_path = 'dataset/real demo/cube_cylinder_300/train_300_cube_cylinder_real_noSmall_slide'
        # with open(data_path, 'rb') as f:
        #     dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])
        
        cotrain_path = 'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal_noSmall'
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)

        # cotrain_path = 'dataset/mixed_object/train_40k_mixed_noSmall_slide'
        # with open(cotrain_path, 'rb') as f:
        #     dataset_cotrain = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        action_cotrain = dataset_cotrain['actions'][:,0]
        dataset_cotrain['actions'] = np.expand_dims(action_cotrain.copy(), axis=1)

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1

        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)
    
    elif args.training_selection == 96:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        # data_path = 'dataset/real demo/cube_cylinder/train_100_cube_cylinder_real_rotation'
        # with open(data_path, 'rb') as f:
        #     dataset = pickle.load(f)
        data_path = 'dataset/real demo/cube_cylinder_200/train_200_cube_cylinder_real_rotation_work'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)
        # with open(f'dataset/real demo/cube_cylinder_300/train_300_cube_cylinder_real_noSmall_rotation', 'rb') as f:
        #     dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])
        
        cotrain_path = 'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_rotation_noObjVel_forReal_noSmall'
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)

        # cotrain_path = 'dataset/mixed_object/train_40k_mixed_noSmall_rotation'
        # with open(cotrain_path, 'rb') as f:
        #     dataset_cotrain = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)

    elif args.training_selection == 97:
        # Sliding
        # + network: 1024
        # + Norm
        # 10k demo
        # + three cylinder
        # + remove obs velocity in obs
        # + for real
        # + add noise
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/three_cylinder/train_10k_three_cylinder_forReal_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: three cylinder only sliding without object velocity")
        print("train_10k_three_cylinder_forReal_noSmall_slide")

    elif args.training_selection == 98:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + three cylinder
        # + remove obs velocity in obs
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/three_cylinder/train_10k_three_cylinder_forReal_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_10k_three_cylinder_forReal_noSmall_rotation")

    elif args.training_selection == 99:
        # finetune
        # cube cylinder

        with open(f'dataset/real demo/cube_cylinder_200/train_200_cube_cylinder_real_slide_work', 'rb') as f:
            dataset = pickle.load(f)

        with open(f'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal_noSmall', 'rb') as f:
            dataset_4_norm = pickle.load(f)

        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_4_norm['observations'][:, -1] = abs(dataset_4_norm['observations'][:, -1])

        goal_mean = np.mean(dataset_4_norm['desired_goals'], axis=0)
        goal_std = np.std(dataset_4_norm['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset_4_norm['observations'], axis=0)
        obs_std = np.std(dataset_4_norm['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1

        print("Fine tune sliding")

    elif args.training_selection == 100:
        with open(f'dataset/real demo/cube_cylinder_200/train_200_cube_cylinder_real_rotation_work', 'rb') as f:
            dataset = pickle.load(f)

        with open(f'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_rotation_noObjVel_forReal_noSmall', 'rb') as f:
            dataset_4_norm = pickle.load(f)

        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_4_norm['observations'][:, -1] = abs(dataset_4_norm['observations'][:, -1])

        goal_mean = np.mean(dataset_4_norm['desired_goals'], axis=0)
        goal_std = np.std(dataset_4_norm['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset_4_norm['observations'], axis=0)
        obs_std = np.std(dataset_4_norm['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1

        print("Fine tune rotation")
    
    elif args.training_selection == 101:

        # data_path = 'dataset/real demo/three_cylinder/train_200_three_cylinder_real_noSmall_slide'
        data_path = 'dataset/real demo/three_cylinder_100/train_100_three_cylinder_real_noSmall_slide'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        cotrain_path = 'dataset/three_cylinder/train_10k_three_cylinder_forReal_noSmall_slide'
        
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)
        
        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        action_cotrain = dataset_cotrain['actions'][:,0]
        dataset_cotrain['actions'] = np.expand_dims(action_cotrain.copy(), axis=1)

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1

        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)
    
    elif args.training_selection == 102:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")

        # data_path = 'dataset/real demo/three_cylinder/train_200_three_cylinder_real_noSmall_rotation'
        data_path = 'dataset/real demo/three_cylinder_100/train_100_three_cylinder_real_noSmall_rotation'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        cotrain_path = 'dataset/three_cylinder/train_10k_three_cylinder_forReal_noSmall_rotation'
        
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)

    elif args.training_selection == 103:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + monkey
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/monkey/train_10k_monkey_forReal_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_10k_monkey_forReal_noSmall_slide")

    elif args.training_selection == 104:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + monkey
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/monkey/train_10k_monkey_forReal_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_10k_monkey_forReal_noSmall_rotation")

    elif args.training_selection == 105:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + chuan
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/chuan/train_chuan_forReal_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_chuan_forReal_noSmall_slide")

    elif args.training_selection == 106:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + chuan
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/chuan/train_chuan_forReal_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_chuan_forReal_noSmall_rotation")

    elif args.training_selection == 107:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + star new
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/star_new/train_star_forReal_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_star_forReal_noSmall_slide")

    elif args.training_selection == 108:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + star new
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/star_new/train_star_forReal_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_star_forReal_noSmall_rotation")

    elif args.training_selection == 109:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + hexagon
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        # 109_2
        # with open(f'dataset/real demo/hexagon/train_200_hexagon_forReal_noSmall_slide', 'rb') as f:
        #     dataset = pickle.load(f)
        # 109_1, 1009
        with open(f'dataset/hexagon/train_hexagon_forReal_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_200_hexagon_forReal_noSmall_slide")

    elif args.training_selection == 110:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + hexagon
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/hexagon/train_hexagon_forReal_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_hexagon_forReal_noSmall_rotation")

    elif args.training_selection == 111:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube_2_5
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_2_5/train_cube_2_5_forReal_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_cube_2_5_forReal_noSmall_slide")

    elif args.training_selection == 112:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube_2_5
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_2_5/train_cube_2_5_forReal_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_cube_2_5_forReal_noSmall_rotation")

    elif args.training_selection == 113:

        # data_path = 'dataset/real demo/hexagon/train_200_hexagon_forReal_noSmall_slide'
        data_path = 'dataset/real demo/hexagon_100/train_100_hexagon_forReal_noSmall_slide'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        cotrain_path = 'dataset/hexagon/train_hexagon_forReal_noSmall_slide'
        
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)
        
        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        action_cotrain = dataset_cotrain['actions'][:,0]
        dataset_cotrain['actions'] = np.expand_dims(action_cotrain.copy(), axis=1)

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1

        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)
    
    elif args.training_selection == 114:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")

        # data_path = 'dataset/real demo/hexagon/train_200_hexagon_forReal_noSmall_rotation'
        data_path = 'dataset/real demo/hexagon_100/train_100_hexagon_forReal_noSmall_rotation'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        cotrain_path = 'dataset/hexagon/train_hexagon_forReal_noSmall_rotation'
        
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)

    elif args.training_selection == 115:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + hexagon
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        # with open(f'dataset/real demo/hexagon/train_200_hexagon_forReal_noSmall_slide', 'rb') as f:
        #     dataset = pickle.load(f)
        with open(f'dataset/real demo/hexagon_100/train_100_hexagon_forReal_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_100_hexagon_forReal_noSmall_slide")

    elif args.training_selection == 116:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + hexagon
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        # with open(f'dataset/real demo/hexagon/train_200_hexagon_forReal_noSmall_rotation', 'rb') as f:
        #     dataset = pickle.load(f)
        with open(f'dataset/real demo/hexagon_100/train_100_hexagon_forReal_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_100_hexagon_forReal_noSmall_rotation")

    elif args.training_selection == 117:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + three cylinder
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        # with open(f'dataset/real demo/three_cylinder/train_200_three_cylinder_real_noSmall_slide', 'rb') as f:
        #     dataset = pickle.load(f)
        with open(f'dataset/real demo/three_cylinder_100/train_100_three_cylinder_real_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)


        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_100_three_cylinder_real_noSmall_slide")

    elif args.training_selection == 118:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + three cylinder
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        # with open(f'dataset/real demo/three_cylinder/train_200_three_cylinder_real_noSmall_rotation', 'rb') as f:
        #     dataset = pickle.load(f)
        with open(f'dataset/real demo/three_cylinder_100/train_100_three_cylinder_real_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_100_three_cylinder_real_noSmall_rotation")

    elif args.training_selection == 119:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube cylinder
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/real demo/cube_cylinder_200/train_200_cube_cylinder_real_slide_work', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_200_cube_cylinder_real_slide_work")

    elif args.training_selection == 120:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + cube cylinder
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/real demo/cube_cylinder_200/train_200_cube_cylinder_real_rotation_work', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_200_cube_cylinder_real_rotation_work")

    elif args.training_selection == 121:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + star
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/real demo/star/train_star_200_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_star_200_noSmall_slide")

    elif args.training_selection == 122:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + star
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/real demo/star/train_star_200_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_star_200_noSmall_rotation")


    elif args.training_selection == 123:
        data_path = 'dataset/real demo/star/train_star_200_noSmall_slide'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        cotrain_path = 'dataset/star_new/train_star_forReal_noSmall_slide'
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)
        
        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        action_cotrain = dataset_cotrain['actions'][:,0]
        dataset_cotrain['actions'] = np.expand_dims(action_cotrain.copy(), axis=1)

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1

        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)
    
    elif args.training_selection == 124:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")

        data_path = 'dataset/real demo/star/train_star_200_noSmall_rotation'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        cotrain_path = 'dataset/star_new/train_star_forReal_noSmall_rotation'
        
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)
    
    elif args.training_selection == 125:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + hexagon
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/hexagon/train_hexagon_forReal_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_hexagon_forReal_noSmall_slide")

    elif args.training_selection == 126:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + hexagon
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/hexagon/train_hexagon_forReal_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print(f"TASK: train_hexagon_forReal_noSmall_rotation")

    elif args.training_selection == 127:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + three cylinder
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/three_cylinder/train_10k_three_cylinder_forReal_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_10k_three_cylinder_forReal_noSmall_slide")

    elif args.training_selection == 128:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + three cylinder
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/three_cylinder/train_10k_three_cylinder_forReal_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_10k_three_cylinder_forReal_noSmall_rotation")

    elif args.training_selection == 129:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + star
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/star/train_star_forReal_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_star_forReal_noSmall_slide")

    elif args.training_selection == 130:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + star
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/star/train_star_forReal_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_star_forReal_noSmall_rotation")

    elif args.training_selection == 131:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + star
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/chuan/train_chuan_forReal_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_chuan_forReal_noSmall_slide")

    elif args.training_selection == 132:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + star
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/chuan/train_chuan_forReal_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_chuan_forReal_noSmall_rotation")

    elif args.training_selection == 133:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        data_path = 'dataset/real demo/cube_cylinder_50/train_50_cube_cylinder_real_slide'
        # data_path = 'dataset/real demo/cube_cylinder/train_100_cube_cylinder_real_slide'
        # data_path = 'dataset/real demo/cube_cylinder_200/train_200_cube_cylinder_real_slide_work'
        # data_path = 'dataset/real demo/cube_cylinder_300/train_300_cube_cylinder_real_noSmall_slide'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])
        
        cotrain_path = 'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_slide_noObjVel_forReal_noSmall'
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        action_cotrain = dataset_cotrain['actions'][:,0]
        dataset_cotrain['actions'] = np.expand_dims(action_cotrain.copy(), axis=1)

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1

        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)
    
    elif args.training_selection == 134:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        data_path = 'dataset/real demo/cube_cylinder_50/train_50_cube_cylinder_real_rotation'
        # data_path = 'dataset/real demo/cube_cylinder/train_100_cube_cylinder_real_rotation'
        # data_path = 'dataset/real demo/cube_cylinder_200/train_200_cube_cylinder_real_rotation_work'
        # data_path = 'dataset/real demo/cube_cylinder_300/train_300_cube_cylinder_real_noSmall_rotation'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])
        
        cotrain_path = 'dataset/cube_cylinder/train_bigSteps_10k_cube_cylinder_rotation_noObjVel_forReal_noSmall'
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)

        # cotrain_path = 'dataset/mixed_object/train_40k_mixed_noSmall_rotation'
        # with open(cotrain_path, 'rb') as f:
        #     dataset_cotrain = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)

    elif args.training_selection == 135:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        data_path = 'dataset/real demo/cube_cylinder_50/train_50_cube_cylinder_real_slide'
        # data_path = 'dataset/real demo/cube_cylinder/train_100_cube_cylinder_real_slide'
        # data_path = 'dataset/real demo/cube_cylinder_200/train_200_cube_cylinder_real_slide_work'
        # data_path = 'dataset/real demo/cube_cylinder_300/train_300_cube_cylinder_real_noSmall_slide'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        cotrain = False

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1

        print("Number of episode: ", num)
        print("data path: ", data_path)
    
    elif args.training_selection == 136:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        data_path = 'dataset/real demo/cube_cylinder_50/train_50_cube_cylinder_real_rotation'
        # data_path = 'dataset/real demo/cube_cylinder/train_100_cube_cylinder_real_rotation'
        # data_path = 'dataset/real demo/cube_cylinder_200/train_200_cube_cylinder_real_rotation_work'
        # data_path = 'dataset/real demo/cube_cylinder_300/train_300_cube_cylinder_real_noSmall_rotation'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        # cotrain_path = 'dataset/mixed_object/train_40k_mixed_noSmall_rotation'
        # with open(cotrain_path, 'rb') as f:
        #     dataset_cotrain = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        cotrain = False

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("data path: ", data_path)

    elif args.training_selection == 137:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + three cylinder
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/real demo/cube_rl_100/train_100_cube_rl_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_100_cube_rl_noSmall_slide")

    elif args.training_selection == 138:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + three cylinder
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/real demo/cube_rl_100/train_100_cube_rl_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_100_cube_rl_noSmall_rotation")

    elif args.training_selection == 139:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + three cylinder
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_rl/train_10k_cube_rl_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_10k_cube_rl_noSmall_slide")

    elif args.training_selection == 140:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + three cylinder
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        with open(f'dataset/cube_rl/train_10k_cube_rl_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_10k_cube_rl_noSmall_rotation")

    elif args.training_selection == 141:

        # data_path = 'dataset/real demo/cube_rl_200/train_200_cube_rl_200_noSmal_slide'
        data_path = 'dataset/real demo/cube_rl_100/train_100_cube_rl_noSmall_slide'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        cotrain_path = 'dataset/cube_rl/train_10k_cube_rl_noSmall_slide'
        
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)
        
        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        action_cotrain = dataset_cotrain['actions'][:,0]
        dataset_cotrain['actions'] = np.expand_dims(action_cotrain.copy(), axis=1)

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1

        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)
    
    elif args.training_selection == 142:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")

        # data_path = 'dataset/real demo/cube_rl_200/train_200_cube_rl_200_noSmal_rotation'
        data_path = 'dataset/real demo/cube_rl_100/train_100_cube_rl_noSmall_rotation'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        cotrain_path = 'dataset/cube_rl/train_10k_cube_rl_noSmall_rotation'
        
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)

    elif args.training_selection == 143:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        # data_path = 'dataset/real demo/star_100_origin/train_100_star_real_noSmall_slide'
        data_path = 'dataset/real demo/star_200_origin/train_200_star_real_noSmall_slide'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])
        
        cotrain_path = 'dataset/star/train_star_forReal_noSmall_slide'
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        action_cotrain = dataset_cotrain['actions'][:,0]
        dataset_cotrain['actions'] = np.expand_dims(action_cotrain.copy(), axis=1)

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1

        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)
    
    elif args.training_selection == 144:
        print(f"Sliding for real - network layer dimension: {args.layer_dim}")
        # data_path = 'dataset/real demo/star_100_origin/train_100_star_real_noSmall_rotation'
        data_path = 'dataset/real demo/star_200_origin/train_200_star_real_noSmall_rotation'
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f)

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])
        
        cotrain_path = 'dataset/star/train_star_forReal_noSmall_rotation'
        with open(cotrain_path, 'rb') as f:
            dataset_cotrain = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])
        dataset_cotrain['observations'][:, -1] = abs(dataset_cotrain['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)

        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        desired_goal_norm_cotrain = normalize(dataset_cotrain['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm
        dataset_cotrain['desired_goals'] = desired_goal_norm_cotrain

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)

        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        obs_norm_cotrain = normalize(dataset_cotrain['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm
        dataset_cotrain['observations'] = obs_norm_cotrain

        cotrain = True

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Cotrain: ", cotrain)
        print("Cotrain ratio: ", args.cotrain_ratio)
        print("data path: ", data_path)
        print("cotrain path: ", cotrain_path)

    elif args.training_selection == 145:
        # Sliding
        # + network: 1024
        # + Norm
        # + 10k demo
        # + star
        # + remove obs velocity in obs
        # + for real
        print(f"Big step with sliding - network layer dimension: {args.layer_dim}")
        # with open(f'dataset/real demo/star_100_origin/train_100_star_real_noSmall_slide', 'rb') as f:
        #     dataset = pickle.load(f)
        with open(f'dataset/real demo/star_200_origin/train_200_star_real_noSmall_slide', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        action = dataset['actions'][:,0]
        dataset['actions'] = np.expand_dims(action.copy(), axis=1)

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("Layer Dim: ", args.layer_dim)
        print("train_100_star_real_noSmall_slide")

    elif args.training_selection == 146:
        # Isolate rotation part as an individual policy
        # + network: 1024
        # + Norm
        # + 10k demo
        # + star
        # + remove obs velocity in obs
        # + for real
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
        # with open(f'dataset/real demo/star_100_origin/train_100_star_real_noSmall_rotation', 'rb') as f:
        #     dataset = pickle.load(f)
        with open(f'dataset/real demo/star_200_origin/train_200_star_real_noSmall_rotation', 'rb') as f:
            dataset = pickle.load(f)

        # abs last value
        dataset['observations'][:, -1] = abs(dataset['observations'][:, -1])

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        for key in dataset.keys(): 
            print(np.shape(dataset[key]))
        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print(f"TASK: train_100_star_real_noSmall_rotation")
        
    else:
        assert args.training_selection == 131, f"Wrong training index {args.training_selection}"
        # Others
        # Training a slide action that takes mid goal as desired goal

    if cotrain == True:
        data_sampler_cotrain = Data_Sampler(dataset_cotrain, args.device)
    data_sampler = Data_Sampler(dataset, args.device)
    writer = SummaryWriter(f"runs/test")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    from agents.ql_diffusion import Diffusion_QL as Agent
    agent = Agent(state_dim=data_sampler.state_dim,
        action_dim=data_sampler.action_dim,
        max_action=args.max_action,
        device=args.device,
        discount=0.99,
        tau=0.005,
        max_q_backup=False,
        eta=0.,  # BC only
        n_timesteps=args.T,
        lr=args.lr,
        lr_decay=True,
        lr_maxt=args.num_epochs,
        grad_norm=1.0,
        layer_dim=args.layer_dim,
        )
    print(f"Layer dimension: {args.layer_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Add noise: {args.add_noise}")
    if args.layer_dim == 256:
        output_dir = f'models_{args.training_selection}'
    elif args.training_selection == 21 or args.training_selection == 20 or args.training_selection == 27 or args.training_selection == 28:
        output_dir = f'models_{args.training_selection}_{args.layer_dim}_1'
    elif args.training_selection == 22:
        output_dir = f'models_{args.training_selection}_{args.layer_dim}_{args.num_demos}'
    elif args.training_selection == 23:
        output_dir = f'models_{args.training_selection}_{args.num_demos}_{args.h1}_{args.h2}_{args.h3}'
    elif cotrain is True:
        output_dir = f'models_{args.training_selection}_{args.layer_dim}_{args.cotrain_ratio}'
    else:
        output_dir = f'models_{args.training_selection}_{args.layer_dim}'
    
    if os.path.exists(output_dir):
        counter = 1
        new_output_dir = f"{output_dir}_{counter}"
        while os.path.exists(new_output_dir):
            counter += 1
            new_output_dir = f"{output_dir}_{counter}"
        output_dir = new_output_dir

    os.makedirs(output_dir, exist_ok=True)
    training_iters = 0

    if args.training_selection == 11:
        print("Load model_10, actor_1200")
        agent.load_model('models_10', 1200)  #  if load trained model
    elif args.training_selection == 20:
        print("Load model_20, actor_10000")
        agent.load_model('models_20_1024', 10000)  #  if load trained model
    elif args.training_selection == 21:
        print("Load model_21, actor_10000")
        agent.load_model('models_21_1024', 10000)  #  if load trained model
    elif args.training_selection == 28:
        agent.load_model('models_28_1024', 20000)
    elif args.training_selection == 27:
        agent.load_model('models_27_1024', 20000)
    elif args.training_selection == 52:
        agent.load_model('models_48_1024', 20000)
    elif args.training_selection == 99:
        agent.load_model('models_88_1024_1', 1500)
    elif args.training_selection == 100:
        agent.load_model('models_89_1024_1', 1500)

    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    while (training_iters < max_timesteps):
        if cotrain is True:
            loss_metric = agent.train(data_sampler,
                                iterations=args.num_steps_per_epoch,
                                batch_size=args.batch_size,
                                log_writer=writer,
                                replay_buffer_cotrain=data_sampler_cotrain,
                                cotrain=True,
                                cotrain_ratio=args.cotrain_ratio
                                )  
        else:
            loss_metric = agent.train(data_sampler,
                                    iterations=args.num_steps_per_epoch,
                                    batch_size=args.batch_size,
                                    log_writer=writer,
                                    add_noise=args.add_noise,
                                    obs_std=obs_std
                                    )       
        training_iters += args.num_steps_per_epoch
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        print(f"Training iterations: {training_iters}")
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        # print loss
        for key, value in loss_metric.items():
            print(f"{key}: {np.mean(value[-100:])}")

        if curr_epoch % 100 == 0:
            agent.save_model(output_dir, curr_epoch)
        


    writer.close()
