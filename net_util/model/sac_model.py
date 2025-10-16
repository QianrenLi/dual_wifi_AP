import torch as th
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, init_log_std: float = 0.0):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), 
            nn.LayerNorm(hidden), 
            nn.GELU(),
            nn.Linear(hidden, act_dim),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), 
            nn.LayerNorm(hidden), 
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        self.critic_target = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), 
            nn.LayerNorm(hidden), 
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

        # State-dependent log std per action dim
        self.log_std_net = nn.Sequential(
            nn.Linear(obs_dim, hidden), 
            nn.LayerNorm(hidden), 
            nn.GELU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: th.Tensor):
        actions = self.actor(obs)
        value = self.critic(th.cat([obs, actions], dim=1))

        # Compute state-dependent log_std
        log_std = self.log_std_net(obs)  # Output log_std from state-dependent network
        std = log_std.exp()  # The std is the exp of log_std
        return actions, std, value

    def act(self, obs: th.Tensor):
        mean, std, value = self(obs)
        dist = th.distributions.Normal(mean, std)
        action = dist.rsample()  # reparameterized sample
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        mean, std, value = self(obs)
        dist = th.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return value, log_prob, entropy
