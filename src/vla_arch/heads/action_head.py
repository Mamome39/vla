import math

import torch
from torch import nn


class ActionHead(nn.Module):
    """Maps a backbone latent (B, in_dim) to a trajectory (B, num_steps, 2) of xy waypoints."""

    def __init__(self, in_dim=2048, num_steps=6):
        super().__init__()
        self.in_dim = in_dim
        self.num_steps = num_steps

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, latent: torch.Tensor, traj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MLPActionHead(ActionHead):
    def __init__(self, in_dim=2048, num_steps=6, hidden_dim=512):
        super().__init__(in_dim, num_steps)
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),  # backbone latents have large norms (~160)
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_steps * 2),
        )

    def forward(self, latent):
        return self.mlp(latent.float()).view(-1, self.num_steps, 2)

    def loss(self, latent, traj):
        return (self(latent) - traj).norm(dim=-1).mean()  # L2 displacement (m)


def timestep_embedding(t, dim):
    """Sinusoidal embedding of t in [0, 1]: (B,) -> (B, dim)."""
    freqs = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    args = 1000 * t[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class FlowMatchingActionHead(ActionHead):
    """Conditional flow matching (rectified flow) over the flattened trajectory.

    Train: x0 ~ N(0, I), x1 = normalized GT, xt = (1 - t) x0 + t x1; regress velocity x1 - x0.
    Sample: integrate dx/dt = v(xt, t, latent) from noise at t=0 to t=1 with Euler steps.
    Each of the `num_samples` trajectories starts from a different noise draw.
    """

    def __init__(self, in_dim=2048, num_steps=6, hidden_dim=512, cond_dim=256, t_dim=64, sample_steps=10,
                 num_samples=3):
        super().__init__(in_dim, num_steps)
        self.t_dim = t_dim
        self.sample_steps = sample_steps
        self.num_samples = num_samples
        act_dim = num_steps * 2
        self.cond = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, cond_dim))
        self.net = nn.Sequential(
            nn.Linear(act_dim + t_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, act_dim),
        )
        # Per-coordinate trajectory normalization, so noise and data are on the same scale.
        self.register_buffer("traj_mean", torch.zeros(num_steps, 2))
        self.register_buffer("traj_std", torch.ones(num_steps, 2))

    def fit_normalization(self, traj):
        """Set normalization from training trajectories (N, num_steps, 2)."""
        self.traj_mean.copy_(traj.mean(0))
        self.traj_std.copy_(traj.std(0).clamp_min(0.1))

    def velocity(self, xt, t, cond):
        return self.net(torch.cat([xt, timestep_embedding(t, self.t_dim), cond], dim=-1))

    def loss(self, latent, traj):
        x1 = ((traj - self.traj_mean) / self.traj_std).flatten(1)
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=x1.device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        v = self.velocity(xt, t, self.cond(latent.float()))
        return (v - (x1 - x0)).pow(2).mean()

    @torch.no_grad()
    def forward(self, latent, generator=None, num_samples=None):
        """Returns (B, K, num_steps, 2): K trajectories per latent, one per initial noise."""
        k = num_samples or self.num_samples
        cond = self.cond(latent.float()).repeat_interleave(k, dim=0)  # (B*K, cond_dim)
        x = torch.randn(len(cond), self.num_steps * 2, device=latent.device, generator=generator)
        dt = 1.0 / self.sample_steps
        for i in range(self.sample_steps):
            t = torch.full((len(x),), i * dt, device=x.device)
            x = x + dt * self.velocity(x, t, cond)
        return x.view(len(latent), k, self.num_steps, 2) * self.traj_std + self.traj_mean
