import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger
from normalizer import Normalizer

from agents.diffusion import Diffusion
from agents.model import MLP, Critic
from agents.helpers import EMA
import pickle

def normalize(data, mean, std):
    # Ensure standard deviation is not zero to avoid division by zero error
    std = np.where(std == 0, 1, std)
    return (data - mean) / std

class Diffusion_QL(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 layer_dim=256,
                 include_action_in_obs=False
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device, layer_dim=layer_dim, include_action_in_obs=include_action_in_obs)
        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        # self.critic = Critic(state_dim, action_dim).to(device)
        # self.critic_target = copy.deepcopy(self.critic)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            # self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup

        # self.state_normalizer = Normalizer(state_dim, default_clip_range=5)
        # self.goal_normalizer = Normalizer(self.n_goals, default_clip_range=5)

        ''' Task2 '''
        # checkpoint = torch.load(
        #     '/Users/qiyangyan/Desktop/Training Files/Trained Policy/Training4_2mm_DR/VariableFriction_3_24.pth')
        # state_normalizer_mean = checkpoint["state_normalizer_mean"]
        # self.obs_mean = state_normalizer_mean
        # state_normalizer_std = checkpoint["state_normalizer_std"]
        # self.obs_std = state_normalizer_std
        # self.goal_mean = 0
        # self.goal_std = 1

        ''' Task3 '''
        # checkpoint = torch.load(
        #     '/Users/qiyangyan/Desktop/Training Files/Trained Policy/Training4_2mm_DR/VariableFriction_3_24.pth')
        # state_normalizer_mean = checkpoint["state_normalizer_mean"]
        # self.obs_mean = state_normalizer_mean
        # state_normalizer_std = checkpoint["state_normalizer_std"]
        # self.obs_std = state_normalizer_std
        # with open(f'/Users/qiyangyan/Desktop/Training Files/Real4/Real4/demonstration/VFF-1686demos', 'rb') as f:
        #     dataset = pickle.load(f)
        # self.goal_mean = np.mean(dataset['desired_goals'], axis=0)
        # self.goal_std = np.std(dataset['desired_goals'], axis=0)

        ''' Task4 '''
        self.obs_mean = 0
        self.obs_std = 1
        self.goal_mean = 0
        self.goal_std = 1

        ''' Task5 '''
        # with open(f'/Users/qiyangyan/Desktop/Training Files/Real4/Real4/demonstration/VFF-1817demos', 'rb') as f:
        #     dataset = pickle.load(f)
        # self.obs_mean = np.mean(dataset['observations'], axis=0)
        # self.obs_std = np.std(dataset['observations'], axis=0)
        # self.goal_mean = np.mean(dataset['desired_goals'], axis=0)
        # self.goal_std = np.std(dataset['desired_goals'], axis=0)


        # with open(f'/Users/qiyangyan/Desktop/Training Files/Real4/Real4/demonstration/VFF-1817demos', 'rb') as f:
        #     dataset = pickle.load(f)
        # self.goal_mean = np.mean(dataset['desired_goals'], axis=0)
        # self.goal_std = np.std(dataset['desired_goals'], axis=0)
        # self.obs_mean = np.mean(dataset['observations'], axis=0)
        # self.obs_std = np.std(dataset['observations'], axis=0)

        # print("Check: ", self.goal_mean, self.goal_std)

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        # metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        metric = {'bc_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            """ Q Training """
            # current_q1, current_q2 = self.critic(state, action)

            # if self.max_q_backup:
            #     next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
            #     next_action_rpt = self.ema_model(next_state_rpt)
            #     target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
            #     target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
            #     target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
            #     target_q = torch.min(target_q1, target_q2)
            # else:
            #     next_action = self.ema_model(next_state)
            #     target_q1, target_q2 = self.critic_target(next_state, next_action)
            #     target_q = torch.min(target_q1, target_q2)

            # target_q = (reward + not_done * self.discount * target_q).detach()

            # critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            # self.critic_optimizer.zero_grad()
            # critic_loss.backward()
            # if self.grad_norm > 0:
            #     critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            # self.critic_optimizer.step()

            """ Policy Training """
            bc_loss = self.actor.loss(action, state)
            # new_action = self.actor(state)

            # q1_new_action, q2_new_action = self.critic(state, new_action)
            # if np.random.uniform() > 0.5:
            #     q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            # else:
            #     q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            q_loss = 0.
            actor_loss = bc_loss + self.eta * q_loss
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()


            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    # log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                # log_writer.add_scalar('Actor Loss', actor_loss.item(), self.step)
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                # log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                # log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                # log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            # metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            # metric['ql_loss'].append(q_loss.item())
            # metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            # self.critic_lr_scheduler.step()

        return metric

    # def sample_action(self, state):
    #     state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
    #     state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
    #     with torch.no_grad():
    #         action = self.actor.sample(state_rpt)
    #         q_value = self.critic_target.q_min(state_rpt, action).flatten()
    #         idx = torch.multinomial(F.softmax(q_value), 1)
    #     return action[idx].cpu().data.numpy().flatten()

    def choose_action(self, joint_state, desired_goal, train_mode=False):
        # joint_state_norm = self.state_normalizer.normalize(joint_state)[:8]
        joint_state_norm = normalize(joint_state, self.obs_mean, self.obs_std)
        desired_goal_norm = normalize(desired_goal, self.goal_mean, self.goal_std)
        state = np.concatenate([desired_goal_norm, joint_state_norm])
        # print(np.shape(state))
        action = self.sample_action(state)
        return action, None, None

    def sample_action(self, state):
        # sample one action per state
        # print(state)
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
        # print(action)
        return action.cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            # torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            # torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth', map_location='cpu'))
            # self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth', map_location='cpu'))
            # self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))

    def load_weights_play(self, path):
        # self.load_model('/Users/qiyangyan/Desktop/Training Files/Real4/Real4/models', 2000)
        print("Diffusion model")
        pass

    def set_to_eval_mode(self):
        # self.model.eval()
        # self.actor.eval()
        print("Diffusion model EVAL")
        pass