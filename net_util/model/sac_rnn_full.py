from dataclasses import dataclass
from typing import Optional, Tuple
import contextlib
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ Network ------------------------------- #
class FeatureExtractor(nn.Module):
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.rnn_hidden = hidden
        self.mlp  = nn.Sequential(nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru  = nn.GRUCell(hidden, hidden)
        self.activation = nn.Sequential(nn.LayerNorm(hidden), nn.GELU())

    def init_state(self, bsz: int, device=None): return th.zeros(bsz, self.rnn_hidden, device=device)

    def forward(self, obs: th.Tensor, h: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mlp = self.mlp(obs)
        h1 = self.gru(mlp, h)
        feat = self.activation(h1)                 # features for heads
        return feat, h1
    
    
class Network(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=128, rnn_k=8, rnn_hidden=64,
                 init_log_std=-2.0, log_std_min=-20.0, log_std_max=2.0, scale_log_offset = None):
        super().__init__()
        self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)
        self.fe = FeatureExtractor(obs_dim, hidden)
        self.fe_t = FeatureExtractor(obs_dim, hidden)

        # Actor
        self.mu = nn.Linear(hidden, act_dim)
        self.logstd_head = nn.Linear(hidden, act_dim)
        self.logstd_bias = nn.Parameter(th.full((act_dim,), init_log_std))

        make_q = lambda: nn.Sequential(nn.Linear(hidden + act_dim, 1))
        self.q1, self.q2 = make_q(), make_q()
        self.q1_t, self.q2_t = make_q(), make_q()
        self._hard_sync()
                
        self.scale_log_offset = scale_log_offset

    def _hard_sync(self):
        for t, s in zip(self.q1_t.parameters(), self.q1.parameters()): t.data.copy_(s.data)
        for t, s in zip(self.q2_t.parameters(), self.q2.parameters()): t.data.copy_(s.data)

    @staticmethod
    def _tanh_log_det(u: th.Tensor) -> th.Tensor:
        return 2 * (th.log(th.tensor(2.0, device=u.device)) - u - F.softplus(-2*u))

    def _mean_std(self, feat: th.Tensor):
        norm = feat.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        mu = self.mu(feat / norm) # penultimate normalization
        logstd = th.clamp(self.logstd_bias + self.logstd_head(feat), self.log_std_min, self.log_std_max)
        return mu, logstd.exp()

    # public API used by the trainer
    def init_hidden(self, bsz: int, device=None):
        return self.fe.init_state(bsz, device)

    def encode(self, obs: th.Tensor, h: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.fe(obs, h)
    

    def encode_target(self, obs: th.Tensor, h: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.fe_t(obs, h)

    def sample_from_features(self, feat: th.Tensor, detach_feat_for_actor: bool = True):
        x = feat.detach() if detach_feat_for_actor else feat
        mu, std = self._mean_std(x)
        dist = th.distributions.Normal(mu, std)
        u = dist.rsample()
        a = th.tanh(u)
        logp_n = dist.log_prob(u).sum(-1, True) - self._tanh_log_det(u).sum(-1, True)
        if self.scale_log_offset:
            logp_n = logp_n - self.scale_log_offset
            
        return a, logp_n

    def q(self, feat: th.Tensor, a: th.Tensor):
        x = th.cat([feat, a], -1)
        return self.q1(x), self.q2(x)

    @th.no_grad()
    def target_backup(self, nxt: th.Tensor, h_tp1: th.Tensor, r: th.Tensor, d: th.Tensor, gamma: float, alpha: th.Tensor):
        feat_n, _ = self.encode_target(nxt, h_tp1)
        mu, std = self._mean_std(feat_n)
        dist = th.distributions.Normal(mu, std)
        u = dist.rsample()
        a_n = th.tanh(u)
        logp_n = dist.log_prob(u).sum(-1, True) - self._tanh_log_det(u).sum(-1, True)
        if self.scale_log_offset:
            logp_n = logp_n - self.scale_log_offset
            
        qmin = th.min(self.q1_t(th.cat([feat_n, a_n], -1)), self.q2_t(th.cat([feat_n, a_n], -1)))
        return r + (1 - d) * gamma * (qmin - alpha * logp_n)

    @contextlib.contextmanager
    def critics_frozen(self):
        flags = [p.requires_grad for p in self.q1.parameters()] + [p.requires_grad for p in self.q2.parameters()]
        for p in list(self.q1.parameters()) + list(self.q2.parameters()): p.requires_grad_(False)
        try: yield
        finally:
            for p, f in zip(list(self.q1.parameters()) + list(self.q2.parameters()), flags): p.requires_grad_(f)

    # param groups
    def actor_parameters(self):
        return list(self.mu.parameters()) + list(self.logstd_head.parameters()) + [self.logstd_bias]
    def critic_parameters(self):
        return list(self.q1.parameters()) + list(self.q2.parameters()) + list(self.fe.parameters())
