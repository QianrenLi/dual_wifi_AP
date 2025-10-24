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
    def __init__(self, obs_dim: int, act_dim: int, hidden=128, belief_dim = 1,
                 init_log_std=-2.0, log_std_min=-20.0, log_std_max=2.0, scale_log_offset = None):
        super().__init__()
        self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)
        
        # FeatureExtractor
        self.fe = FeatureExtractor(obs_dim + belief_dim, hidden)
        self.fe_t = FeatureExtractor(obs_dim + belief_dim, hidden)
        
        self.belief_rnn = FeatureExtractor(obs_dim, hidden)
        self.belief_head = nn.Linear(hidden, 1)

        # Actor
        self.mu = nn.Linear(hidden, act_dim)
        self.logstd_head = nn.Linear(hidden, act_dim)
        self.logstd_bias = nn.Parameter(th.full((act_dim,), init_log_std))
        
        # Critic
        def _make_q():
            return nn.Sequential(
                nn.Linear(hidden + act_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        self.q1, self.q2 = _make_q(), _make_q()
        self.q1_t, self.q2_t = _make_q(), _make_q()

        self._hard_sync()
        self.scale_log_offset = scale_log_offset

    def _hard_sync(self):
        for t, s in zip(self.q1_t.parameters(), self.q1.parameters()): t.data.copy_(s.data)
        for t, s in zip(self.q2_t.parameters(), self.q2.parameters()): t.data.copy_(s.data)
        for t, s in zip(self.fe_t.parameters(), self.fe.parameters()): t.data.copy_(s.data)

    @staticmethod
    def _tanh_log_det(u: th.Tensor) -> th.Tensor:
        return 2 * (th.log(th.tensor(2.0, device=u.device)) - u - F.softplus(-2*u))

    def _mean_std(self, feat: th.Tensor):
        norm = feat.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        feat = feat / norm # penultimate normalization
        mu = self.mu(feat) 
        logstd = th.clamp(self.logstd_bias + self.logstd_head(feat), self.log_std_min, self.log_std_max)
        return mu, logstd.exp()
    
    def belief_predict(self, obs:th.Tensor, belief_h: th.Tensor):
        feat, nxt_bh = self.belief_rnn(obs, belief_h)
        return self.belief_head(feat), nxt_bh

    # public API used by the trainer
    def init_hidden(self, bsz: int, device=None): 
        return self.fe.init_state(bsz, device), self.belief_rnn.init_state(bsz, device)
    
    def encode(self, obs: th.Tensor, belief: th.Tensor, h: th.Tensor): 
        return self.fe( th.cat([obs, belief], -1) , h)
    
    def encode_target(self, obs: th.Tensor, belief: th.Tensor, h: th.Tensor): 
        return self.fe_t( th.cat([obs, belief], -1) , h)

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
    def target_backup(self, nxt: th.Tensor, belief: th.Tensor, h_tp1: th.Tensor, r: th.Tensor, d: th.Tensor, gamma: float, alpha: th.Tensor):
        feat_n, _ = self.encode_target(nxt, belief, h_tp1)
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
    def critics_frozen(self, freeze_encoder: bool = True):
        ps = list(self.q1.parameters()) + list(self.q2.parameters())
        flags = [p.requires_grad for p in ps]
        for p in ps: p.requires_grad_(False)

        fe_flags = None
        if freeze_encoder:
            fe_flags = [p.requires_grad for p in self.fe.parameters()]
            for p in self.fe.parameters(): p.requires_grad_(False)
        try:
            yield
        finally:
            for p, f in zip(ps, flags): p.requires_grad_(f)
            if freeze_encoder:
                for p, f in zip(self.fe.parameters(), fe_flags): p.requires_grad_(f)


    # param groups
    def actor_parameters(self):
        return list(self.mu.parameters()) + list(self.logstd_head.parameters()) + [self.logstd_bias]
    def critic_parameters(self):
        return list(self.q1.parameters()) + list(self.q2.parameters()) + list(self.fe.parameters())
    def belief_parameteres(self):
        return list(self.belief_rnn.parameters()) + list(self.belief_head.parameters())
