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
    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(args.device)

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
        print(f"Big step only Rotation - network layer dimension: {args.layer_dim}")
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
        # Action with slide stop indicator
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


    else:
        assert args.training_selection == 29, f"Wrong training index {args.training_selection}"
        # Others

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
    if args.layer_dim == 256:
        output_dir = f'models_{args.training_selection}'
    elif args.training_selection == 21 or args.training_selection == 20:
        output_dir = f'models_{args.training_selection}_{args.layer_dim}_1'
    elif args.training_selection == 22:
        output_dir = f'models_{args.training_selection}_{args.layer_dim}_{args.num_demos}'
    elif args.training_selection == 23:
        output_dir = f'models_{args.training_selection}_{args.num_demos}_{args.h1}_{args.h2}_{args.h3}'
    else:
        output_dir = f'models_{args.training_selection}_{args.layer_dim}'
    os.makedirs(output_dir, exist_ok=True)
    training_iters = 0

    if args.training_selection == 11:
        print("Load model_10, actor_1200")
        agent.load_model('models_10', 1200)  #  if load trained model
    elif args.training_selection == 20:
        print("Load model_20, actor_10000")
        agent.load_model('models_20_1024', 10000)  #  if load trained model
    # elif args.training_selection == 21:
    #     print("Load model_21, actor_10000")
    #     agent.load_model('models_21_1024', 10000)  #  if load trained model

    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    while (training_iters < max_timesteps):
        loss_metric = agent.train(data_sampler,
                                  iterations=args.num_steps_per_epoch,
                                  batch_size=args.batch_size,
                                  log_writer=writer)       
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
