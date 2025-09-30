import torch as th
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, init_log_std: float = 0.0):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), 
            nn.LayerNorm(hidden), 
            nn.GELU(),
            nn.Linear(hidden, hidden), 
            nn.LayerNorm(hidden), 
            nn.GELU(),
        )
        self.action_mean = nn.Sequential(
            nn.Linear(hidden, act_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden, 1)
        # Global (state-independent) log std per action dim
        self.log_std = nn.Parameter(th.ones(act_dim) * init_log_std)

    def forward(self, obs: th.Tensor):
        z = self.backbone(obs)
        mean = self.action_mean(z)
        value = self.value_head(z).squeeze(-1)
        std = self.log_std.exp()
        return mean, std, value

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