import numpy as np
import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):
        if len(time.shape) > 1:
            time = time.squeeze(1)  # added for shaping t from (batch_size, 1) to (batch_size,)
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)

class ResNetBlock(nn.Module):
    def __init__(self, in_features, hidden_dim, dropout_rate=0.1):
        super(ResNetBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, hidden_dim)
        )

    def forward(self, x):
        identity = x
        out = self.layer(x)
        out += identity
        return out

class LN_Resnet(nn.Module):
    def __init__(self, state_dim, action_dim, device, t_dim=16, hidden_size=256, dropout_rate=0.1):
        super(LN_Resnet, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )
        input_dim = state_dim + action_dim + t_dim

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        )
        self.resnet_block1 = ResNetBlock(hidden_size, hidden_size, dropout_rate)
        self.resnet_block2 = ResNetBlock(hidden_size, hidden_size, dropout_rate)
        self.resnet_block3 = ResNetBlock(hidden_size, hidden_size, dropout_rate)
        self.output_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, x, time, state):
        if len(time.shape) > 1:
            time = time.squeeze(1)  # added for shaping t from (batch_size, 1) to (batch_size,)
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.input_layer(x)
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        x = self.resnet_block3(x)
        x = self.output_layer(x)
        return x



blk = lambda ic, oc: nn.Sequential(
    nn.GroupNorm(32, num_channels=ic),
    nn.SiLU(),
    nn.Conv2d(ic, oc, 3, padding=1),
    nn.GroupNorm(32, num_channels=oc),
    nn.SiLU(),
    nn.Conv2d(oc, oc, 3, padding=1),
)

class Unet(nn.Module):
    def __init__(self, 
        n_channel: int,
        D: int = 128,
        device: torch.device = torch.device("cpu"),
        ) -> None:
        super(Unet, self).__init__()
        self.device = device

        self.freqs = torch.exp(
            -math.log(10000) * torch.arange(start=0, end=D, dtype=torch.float32) / D
        )

        self.down = nn.Sequential(
            *[
                nn.Conv2d(n_channel, D, 3, padding=1),
                blk(D, D),
                blk(D, 2 * D),
                blk(2 * D, 2 * D),
            ]
        )

        self.time_downs = nn.Sequential(
            nn.Linear(2 * D, D),
            nn.Linear(2 * D, D),
            nn.Linear(2 * D, 2 * D),
            nn.Linear(2 * D, 2 * D),
        )

        self.mid = blk(2 * D, 2 * D)

        self.up = nn.Sequential(
            *[
                blk(2 * D, 2 * D),
                blk(2 * 2 * D, D),
                blk(D, D),
                nn.Conv2d(2 * D, 2 * D, 3, padding=1),
            ]
        )
        self.last = nn.Conv2d(2 * D + n_channel, n_channel, 3, padding=1)

    def forward(self, x, t) -> torch.Tensor:
        # time embedding
        args = t.float() * self.freqs[None].to(t.device)
        t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1).to(x.device)

        x_ori = x

        # perform F(x, t)
        hs = []
        for idx, layer in enumerate(self.down):
            if idx % 2 == 1:
                x = layer(x) + x
            else:
                x = layer(x)
                x = F.interpolate(x, scale_factor=0.5)
                hs.append(x)

            x = x + self.time_downs[idx](t_emb)[:, :, None, None]

        x = self.mid(x)

        for idx, layer in enumerate(self.up):
            if idx % 2 == 0:
                x = layer(x) + x
            else:
                x = torch.cat([x, hs.pop()], dim=1)
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                x = layer(x)

        x = self.last(torch.cat([x, x_ori], dim=1))

        return x
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
    
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=0.02):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    

class MLPNetwork(nn.Module):
    """Simple multi-layer perceptron network."""
    def __init__(self, input_dim, hidden_dim=100, num_hidden_layers=1, output_dim=1, dropout_rate=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim)] + \
                 [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers - 1)] + \
                 [nn.Linear(hidden_dim, output_dim)]
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.Mish()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # Apply activation and dropout to all but the last layer
                x = self.act(x)
                x = self.dropout(x)
        return x

    def get_device(self, device: torch.device):
        self._device = device
        self.layers.to(device)
    
    def get_params(self):
        return self.layers.parameters()


class ConsistencyTrajectoryNetwork(nn.Module):
    def __init__(self, x_dim, hidden_dim, time_embed_dim, cond_dim, cond_mask_prob,
                 num_hidden_layers=1, output_dim=1, dropout_rate=0.0, cond_conditional=True):
        super().__init__()
        self.embed_t = GaussianFourierProjection(time_embed_dim)
        self.embed_s = GaussianFourierProjection(time_embed_dim)
        self.cond_mask_prob = cond_mask_prob
        self.cond_conditional = cond_conditional
        input_dim = time_embed_dim * 2 + x_dim + (cond_dim if cond_conditional else 0)
        self.mlp = MLPNetwork(input_dim, hidden_dim, num_hidden_layers, output_dim, dropout_rate)

    def forward(self, x, cond, t, s):
        t = t.view(-1, 1)
        s = s.view(-1, 1)

        embed_t = self.embed_t(t).squeeze(1)
        embed_s = self.embed_s(s).squeeze(1)
        if embed_s.shape[0] != x.shape[0]:
            embed_s = einops.repeat(embed_s, '1 d -> (1 b) d', b=x.shape[0])
        if embed_t.shape[0] != x.shape[0]:
            embed_t = einops.repeat(embed_t, '1 d -> (1 b) d', b=x.shape[0])
        x = torch.cat([x, cond, embed_s, embed_t], dim=-1) if self.cond_conditional else torch.cat([x, embed_s, embed_t], dim=-1)
        return self.mlp(x)


def rearrange_for_batch(x, batch_size):
    """Utility function to repeat the tensor for the batch size."""
    return x.expand(batch_size, -1)

    def get_params(self):
        return self.parameters()

class Discriminator(nn.Module):
    """ MLPNetwork with sigmoid activation"""
    def __init__(self, input_dim, hidden_dim=100, num_hidden_layers=1, output_dim=1, dropout_rate=0.0):
        super().__init__()
        self.mlp = MLPNetwork(input_dim, hidden_dim, num_hidden_layers, output_dim, dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.mlp(x))

    def get_device(self, device: torch.device):
        self._device = device
        self.mlp.to(device)
    
    def get_params(self):
        return self.mlp.parameters()