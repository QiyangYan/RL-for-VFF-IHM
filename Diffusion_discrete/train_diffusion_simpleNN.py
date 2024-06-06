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
    parser.add_argument('--device', default=0, type=int)  # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--max_action", default=1., type=float)
    parser.add_argument("--num_epochs", default=10000, type=int)
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--training_selection", default=None, type=int)
    parser.add_argument("--layer_dim", default=256, type=int)
    parser.add_argument("--num_demos", default=0, type=int)
    parser.add_argument("--h1", default=4, type=int)
    parser.add_argument("--h2", default=2, type=int)
    parser.add_argument("--h3", default=2, type=int)
    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(args.device)

    if args.training_selection == 1:
        # Sliding
        # + network: 1024
        # + Norm
        # + 50k demo
        # + No DR
        # + Isolated End Indicator
        # + 20000
        print(f"Big step with slide stop indicator - network layer dimension: {args.layer_dim}")
        with open(f'/Users/qiyangyan/Desktop/Diffusion/Demonstration/bigSteps_50kdemos_slide_endIndicator', 'rb') as f:
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
        print(f"TASK: bigSteps_50kdemos_slide_endIndicator")

    elif args.training_selection == 2:
        # 3 dimensional action
        # new structure
        print(f"Big step with slide - network layer dimension: {args.layer_dim}")
        with open(f'/rds/general/user/qy320/home/diffusion/dataset/bigSteps_10k_demos_random_slide', 'rb') as f:
            dataset = pickle.load(f)

        goal_mean = np.mean(dataset['desired_goals'], axis=0)
        goal_std = np.std(dataset['desired_goals'], axis=0)
        desired_goal_norm = normalize(dataset['desired_goals'], goal_mean, goal_std)
        dataset['desired_goals'] = desired_goal_norm

        obs_mean = np.mean(dataset['observations'], axis=0)
        obs_std = np.std(dataset['observations'], axis=0)
        obs_norm = normalize(dataset['observations'], obs_mean, obs_std)
        dataset['observations'] = obs_norm

        assert len(dataset['actions']) == len(dataset['terminals']), "length doesn't match"
        # Append the new column to each array in the list
        action_with_end_indicator = np.array([np.append(arr, new_col) for arr, new_col in zip(dataset['actions'], dataset['terminals'])])
        dataset['actions'] = action_with_end_indicator
        print("Updated list of arrays:")
        print(np.shape(action_with_end_indicator))

        for key in dataset.keys():
            print(np.shape(dataset[key]))

        num = 0
        for i, item in enumerate(dataset['terminals']):
            if item:
                num += 1
        print("Number of episode: ", num)
        print("Number of demos: ", args.num_demos)
        print("TASK: bigSteps_10k_demos_random_slide with end indicator")

    elif args.training_selection == 3:
        # only terminals
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

    else:
        assert args.training_selection == 4, f"Wrong training index {args.training_selection}"
        # Others
        # Training a slide action that takes mid goal as desired goal

    data_sampler = Data_Sampler(dataset, args.device)
    writer = SummaryWriter(f"runs/test")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    from agents.ql_diffusion import Diffusion_QL as Agent

    agent = Agent(state_dim=data_sampler.state_dim,
                  action_dim=data_sampler.action_dim-2,
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
    if args.layer_dim == 256:
        output_dir = f'models_{args.training_selection}'
    elif args.training_selection == 21 or args.training_selection == 20 or args.training_selection == 27 or args.training_selection == 28:
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
        agent.load_model('models_10', 1200)  # if load trained model
    elif args.training_selection == 20:
        print("Load model_20, actor_10000")
        agent.load_model('models_20_1024', 10000)  # if load trained model
    elif args.training_selection == 21:
        print("Load model_21, actor_10000")
        agent.load_model('models_21_1024', 10000)  # if load trained model
    elif args.training_selection == 28:
        agent.load_model('models_28_1024', 20000)
    elif args.training_selection == 27:
        agent.load_model('models_27_1024', 20000)
    elif args.training_selection == 52:
        agent.load_model('models_48_1024', 20000)

    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    while (training_iters < max_timesteps):
        loss_metric = agent.train_diff_simpleNN(data_sampler,
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