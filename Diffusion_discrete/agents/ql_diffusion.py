import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from agents.diffusion import Diffusion
# from agents.diffusion_discrete import Diffusion
from agents.model import MLP, Critic
from agents.helpers import EMA
from agents.simple_nn import SimpleNN

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
                 # discrete_mid_layer_dim=128,
                 # discrete_out_layer_dim=64,
                 ):

        self.device = device
        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device, layer_dim=layer_dim)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # TODO: two mroe actors with two more optimiser and split actions, add loss term (cross entropy), dim is number of discrete classes
        # state = obs + desired_goals
        # control mode prediction
        self.actor_discrete_1 = SimpleNN(input_size=state_dim, num_classes=4).to(device)
        self.actor_discrete_1_optimizer = torch.optim.Adam(self.actor_discrete_1.parameters(), lr=0.001)

        # slide end prediction
        self.actor_discrete_2 = SimpleNN(input_size=state_dim, num_classes=2).to(device)
        self.actor_discrete_2_optimizer = torch.optim.Adam(self.actor_discrete_2.parameters(), lr=0.001)

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

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def train_diff_simpleNN(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        # metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        metric = {'bc_loss': [], 'mode_loss': [], 'end_indicator_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            # state, action, next_state, not_done = replay_buffer.sample(batch_size)
            state, action = replay_buffer.sample(batch_size)

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
            action_con = torch.tensor(np.expand_dims(action[:, 0].clone().detach().cpu().numpy(), axis=1)).to(self.device)
            action_mode = action[:, 1].clone().detach().to(self.device)
            action_end_indicator = action[:, 2].clone().detach().to(self.device)

            # diffusion
            bc_loss = self.actor.loss(action_con, state)
            q_loss = 0.
            actor_loss = bc_loss + self.eta * q_loss

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm,
                                                            norm_type=2)
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            """ Discrete Networks """
            # mode
            d1_output = self.actor_discrete_1(state)
            d1_loss = self.actor_discrete_1.loss(d1_output, action_mode)
            d1_loss.backward()
            self.actor_discrete_1_optimizer.step()

            # end indicator
            d2_output = self.actor_discrete_1(state)
            d2_loss = self.actor_discrete_1.loss(d2_output, action_end_indicator)
            d2_loss.backward()
            self.actor_discrete_2_optimizer.step()

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    # log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                # log_writer.add_scalar('Actor Loss', actor_loss.item(), self.step)
                log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                log_writer.add_scalar('Mode Loss', d1_loss.item(), self.step)
                log_writer.add_scalar('End Indicator Loss', d2_loss.item(), self.step)
                # log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                # log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                # log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            # metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            metric['mode_loss'].append(d1_loss.item())
            metric['end_indicator_loss'].append(d2_loss.item())
            # metric['ql_loss'].append(q_loss.item())
            # metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            # self.critic_lr_scheduler.step()

        return metric

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):

        # metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        metric = {'bc_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch
            # state, action, next_state, not_done = replay_buffer.sample(batch_size)
            state, action = replay_buffer.sample(batch_size)

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
            torch.save(self.actor_discrete_1.state_dict(), f'{dir}/actor_discrete_1_{id}.pth')
            torch.save(self.actor_discrete_2.state_dict(), f'{dir}/actor_discrete_2_{id}.pth')
            # torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.actor_discrete_1.state_dict(), f'{dir}/actor_discrete_1_.pth')
            torch.save(self.actor_discrete_2.state_dict(), f'{dir}/actor_discrete_2_.pth')
            # torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth', map_location='cpu'))
            # self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth', map_location='cpu'))
            # self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))


