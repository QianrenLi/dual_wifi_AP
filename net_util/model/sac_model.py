import torch as th
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128, init_log_std: float = -2.0, log_std_min: float = -20, log_std_max: float = 2.0, scale_log_offset = 0):
        super().__init__()
        # ----- Actor -----
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, act_dim),
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
            nn.Linear(obs_dim, act_dim),
        )
        self.log_std_bias = nn.Parameter(th.full((act_dim,), init_log_std))

        # Parameters for constraining log_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.scale_log_offset = scale_log_offset

    def forward(self, obs: th.Tensor):
        """
        Returns:
            actions: (B, act_dim)
            std:     (B, act_dim)
            value:   (B, 1)  # min(Q1, Q2) at (obs, actions)
        """
        actions = self.actor(obs)
        log_std = self.log_std_net(obs)
        
        # Ensure the log_std is within the min/max bounds (extra safety clamp)
        log_std = self.log_std_bias + self.log_std_net(obs)
        log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)

        std = log_std.exp()  # Convert log_std to standard deviation
        return actions, std, 0

    @staticmethod
    def _tanh_log_det_jacobian(u: th.Tensor) -> th.Tensor:
        # Stable: log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))
        return 2.0 * (th.log(th.tensor(2.0, device=u.device)) - u - th.nn.functional.softplus(-2.0 * u))

    def act(self, obs: th.Tensor):
        """
        Returns:
            a:        squashed action in (-1, 1), shape (B, act_dim)
            log_prob: log π(a|s) with tanh correction, shape (B, 1)
            value:    min-Q estimate placeholder (unchanged here)
        """
        mean, std, value = self(obs)                   # mean = actor(obs), std = exp(log_std)
        dist = th.distributions.Normal(mean, std)

        u = dist.rsample()                             # pre-squash action (reparameterized)
        a = th.tanh(u)                                 # bounded action

        # base log prob under N(mean, std): sum over action dims
        logp_u = dist.log_prob(u).sum(dim=-1, keepdim=True)

        # change-of-variables correction: - sum log |det J_tanh(u)|
        # where log|det J| = sum_i log(1 - tanh(u_i)^2).
        # Using stable closed form:
        log_det = self._tanh_log_det_jacobian(u).sum(dim=-1, keepdim=True)

        log_prob = logp_u - log_det

        # If you later scale a to env bounds with a = scale * a + bias,
        # subtract sum(log|scale|) here via self.scale_log_offset:
        if self.scale_log_offset:
            log_prob = log_prob - self.scale_log_offset

        return a, log_prob, value