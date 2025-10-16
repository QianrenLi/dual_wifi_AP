import torch as th
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, init_log_std: float = 0.0, log_std_min: float = -10, log_std_max: float = 2.0):
        super().__init__()
        # ----- Actor -----
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),  # Output between -1 and 1 for the actions
        )

        # ----- Twin Critics -----
        self.critic1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.critic2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # ----- Twin Target Critics -----
        self.critic1_target = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.critic2_target = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # ----- State-dependent log std per action dim -----
        self.log_std_net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, act_dim),
        )

        # Parameters for constraining log_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, obs: th.Tensor):
        """
        Returns:
            actions: (B, act_dim)
            std:     (B, act_dim)
            value:   (B, 1)  # min(Q1, Q2) at (obs, actions)
        """
        actions = self.actor(obs)
        log_std = self.log_std_net(obs)
        
        # Apply tanh and scaling to constrain log_std to the range [-10, 2] or a similar desired range
        log_std = th.tanh(log_std)  # Apply tanh to limit the range to [-1, 1]
        log_std = log_std * (self.log_std_max - self.log_std_min) / 2  # Scale to the target range
        log_std = log_std + (self.log_std_max + self.log_std_min) / 2  # Shift the values to be in the target range

        # Ensure the log_std is within the min/max bounds (extra safety clamp)
        log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)

        std = log_std.exp()  # Convert log_std to standard deviation
        return actions, std, 0

    def act(self, obs: th.Tensor):
        """
        Stochastic action (reparameterized), log_prob, and min-Q value.
        """
        mean, std, value = self(obs)
        dist = th.distributions.Normal(mean, std)
        action = dist.rsample()  # reparameterized sample
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        """
        Evaluate provided actions under current policy.
        Returns:
            value:   min(Q1, Q2) at (obs, actions)
            log_prob, entropy
        """
        # Policy stats for log_prob/entropy
        mean, std, _ = self(obs)
        dist = th.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        # Twin Q for provided actions
        inp = th.cat([obs, actions], dim=-1)
        q1 = self.critic1(inp)
        q2 = self.critic2(inp)
        value = th.minimum(q1, q2)
        return value, log_prob, entropy
