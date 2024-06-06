import pickle  
import argparse  
import torch
import os
from utils.data_sampler import Data_Sampler
from utils import utils
import numpy as np

with open(f'dataset/VFF-1686demos', 'rb') as f:
    dataset = pickle.load(f)
# obs = np.array([item[:8] for item in dataset['observations']])
# dataset['observations'] = obs
goal_mean = np.mean(dataset['desired_goals'], axis=0)
goal_std = np.std(dataset['desired_goals'], axis=0)

def normalize(data, mean, std):
    # Ensure standard deviation is not zero to avoid division by zero error
    std = np.where(std == 0, 1, std)
    return (data - mean) / std

def discretize_action_to_control_mode(action):
    # Your action discretization logic here
    action_norm = (action + 1) / 2
    if 1 / 4 > action_norm >= 0:
        control_mode = 0
        friction_state = 1
    elif 2 / 4 > action_norm >= 1 / 4:
        control_mode = 1
        friction_state = 1
    elif 3 / 4 > action_norm >= 2 / 4:
        control_mode = 2
        friction_state = -1
    else:
        assert 1 > action_norm >= 3 / 4
        control_mode = 3
        friction_state = -1
    return friction_state, control_mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument('--device', default=0, type=int)                       # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--max_action", default=1., type=float)
    parser.add_argument("--num_epochs", default=1000, type=int)
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--T", default=5, type=int)
    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    data_sampler = Data_Sampler(dataset, args.device)

    from agents.ql_diffusion import Diffusion_QL as Agent
    agent = Agent(
        # state_dim=data_sampler.state_dim-2,
        state_dim=data_sampler.state_dim,
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
        )
    output_dir = 'models'
    model_idx = 5430

    agent.load_model(output_dir, model_idx)
    agent.model.eval()
    agent.actor.eval()

    # input concatenation of observation and goal, output action
    for i in range(10):
        obs = dataset['observations'][i]
        # obs = dataset['observations'][i]
        # goal = normalize(dataset['desired_goals'][i], mean=goal_mean, std=goal_std)
        goal = dataset['desired_goals'][i]
        true_action = dataset['actions'][i]

        state = np.concatenate([goal, obs])
        action = agent.sample_action(state)
        _, action_dis = discretize_action_to_control_mode(action[1])
        action[1] = action_dis
        _, action_dis = discretize_action_to_control_mode(true_action[1])
        true_action[1] = action_dis
        print(action, true_action)


