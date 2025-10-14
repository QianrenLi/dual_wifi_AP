import torch as th
import torch.nn as nn
from typing import Any, Dict
from util.trace_collec import flatten_leaves


class Network(nn.Module):
    def __init__(self, obs_dim: int, rnn_obs_dim: int, act_dim: int, device, hidden: int = 128, init_log_std: float = 0.0, rnn_hidden: int = 64):
        super().__init__()
        
        # Backbone for standard observations (non-sequential)
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        # LSTM for sequential observations (rnn_obs)
        self.rnn = nn.LSTM(rnn_obs_dim, rnn_hidden, batch_first=True)

        # Action mean and value heads
        self.action_mean = nn.Linear(hidden + rnn_hidden, act_dim)
        self.value_head = nn.Linear(hidden + rnn_hidden, 1)

        # Global (state-independent) log std per action dim
        self.log_std = nn.Parameter(th.ones(act_dim) * init_log_std)

        # Device setup
        self.device = th.device(device)

    def forward(self, obs: th.Tensor, rnn_obs: th.Tensor):
        # Process non-sequential (static) observations
        z = self.backbone(obs)

        # Process sequential (RNN) observations
        rnn_out, (hn, cn) = self.rnn(rnn_obs)
        rnn_out = rnn_out[:, -1, :]  # Use the output of the last time step

        # Concatenate the outputs from both parts
        combined = th.cat((z, rnn_out), dim=-1)

        # Action and value prediction
        mean = self.action_mean(combined)
        value = self.value_head(combined).squeeze(-1)
        std = self.log_std.exp()

        return mean, std, value

    def act(self, obs_dict: Dict[str, Any]):
        # Flatten static observations
        obs = th.tensor(flatten_leaves(obs_dict), device=self.device).float()

        # Flatten RNN observations
        rnn_obs = th.tensor(flatten_leaves(obs_dict['rnn']), device=self.device).float()

        # Forward pass
        mean, std, value = self(obs, rnn_obs)

        # Sample action
        dist = th.distributions.Normal(mean, std)
        action = dist.rsample()  # Reparameterized sample
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob, value

    def enjoy(self, obs_dict: Dict[str, Any]):
        # Flatten static observations
        obs = th.tensor(flatten_leaves(obs_dict), device=self.device).float()

        # Flatten RNN observations
        rnn_obs = th.tensor(flatten_leaves(obs_dict['rnn']), device=self.device).float()

        # Forward pass
        mean, std, value = self(obs, rnn_obs)

        return mean, std, value

    def evaluate_actions(self, obs_dict: Dict[str, Any], actions: th.Tensor):
        # Flatten static observations
        obs = th.tensor(flatten_leaves(obs_dict), device=self.device).float()

        # Flatten RNN observations
        rnn_obs = th.tensor(flatten_leaves(obs_dict['rnn']), device=self.device).float()

        # Forward pass
        mean, std, value = self(obs, rnn_obs)

        # Compute log probabilities and entropy
        dist = th.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        return value, log_prob, entropy
