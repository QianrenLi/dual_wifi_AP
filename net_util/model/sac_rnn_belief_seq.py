from dataclasses import dataclass
from typing import Optional, Tuple
import contextlib
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------ Feature Extractor (sequence GRU) ------------------------------- #
class FeatureExtractorGRU(nn.Module):
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.mlp  = nn.Sequential(nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru  = nn.GRU(input_size=hidden, hidden_size=hidden, num_layers=1, batch_first=False)
        self.post = nn.Sequential(nn.LayerNorm(hidden), nn.GELU())

    def init_state(self, bsz: int, device=None):
        # shape [num_layers, B, H] = [1, B, H]
        return th.zeros(1, bsz, self.hidden, device=device)

    # step input: obs [B,D]; h0 [1,B,H] -> feat [B,H], h1 same shape as h0
    def forward_step(self, obs: th.Tensor, h0: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        x = self.mlp(obs)                             # [B,H]
        x = x.unsqueeze(0)                            # [1,B,H] as time=1
        y, h1 = self.gru(x, h0)                       # y:[1,B,H], h1:[1,B,H]
        feat = self.post(y.squeeze(0))                # [B,H]
        return feat, h1

    # sequence input: obs [T,B,D]; h0 [1,B,H] -> feat [T,B,H], hT [1,B,H]
    def forward_seq(self, obs_TBD: th.Tensor, h0: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        T, B, D = obs_TBD.shape
        x = self.mlp(obs_TBD.view(T*B, D)).view(T, B, -1)   # [T,B,H]
        y, hT = self.gru(x, h0)                             # y:[T,B,H]
        feat_TB = self.post(y)                              # [T,B,H]
        return feat_TB, hT


class Network(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=128, belief_dim=1,
                 init_log_std=-2.0, log_std_min=-20.0, log_std_max=2.0, scale_log_offset=None):
        super().__init__()
        self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)

        # Encoders
        self.fe   = FeatureExtractorGRU(obs_dim + belief_dim, hidden)
        self.fe_t = FeatureExtractorGRU(obs_dim + belief_dim, hidden)

        self.belief_rnn  = FeatureExtractorGRU(obs_dim, hidden)
        self.belief_head = nn.Linear(hidden, 1)

        # Actor
        self.mu          = nn.Linear(hidden, act_dim)
        self.logstd_head = nn.Linear(hidden, act_dim)
        self.logstd_bias = nn.Parameter(th.full((act_dim,), init_log_std))

        # Critics
        def _make_q():
            return nn.Sequential(
                nn.Linear(hidden + act_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden),          nn.GELU(),
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
        # penultimate normalization (pnorm)
        norm = feat.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        feat = feat / norm
        mu = self.mu(feat)
        logstd = th.clamp(self.logstd_bias + self.logstd_head(feat), self.log_std_min, self.log_std_max)
        return mu, logstd.exp()
    
    # -------- Belief --------
    def belief_predict_step(self, obs_BD: th.Tensor, h0: th.Tensor):
        feat_BH, h1 = self.belief_rnn.forward_step(obs_BD, h0)
        return self.belief_head(feat_BH), h1

    def belief_predict_seq(self, obs_TBD: th.Tensor, h0: th.Tensor):
        feat_TBH, hT = self.belief_rnn.forward_seq(obs_TBD, h0)
        return self.belief_head(feat_TBH), hT                       # [T,B,1], [1,B,H]

    # -------- Public API used by the trainer --------
    def init_hidden(self, bsz: int, device=None):
        return self.fe.init_state(bsz, device), self.belief_rnn.init_state(bsz, device)

    # step encoders
    def encode(self, obs_BD: th.Tensor, belief_B1: th.Tensor, h0: th.Tensor):
        x = th.cat([obs_BD, belief_B1], dim=-1)
        return self.fe.forward_step(x, h0)

    def encode_target(self, obs_BD: th.Tensor, belief_B1: th.Tensor, h0: th.Tensor):
        x = th.cat([obs_BD, belief_B1], dim=-1)
        return self.fe_t.forward_step(x, h0)

    # sequence encoders
    def encode_seq(self, obs_TBD: th.Tensor, belief_TB1: th.Tensor, h0: th.Tensor):
        x = th.cat([obs_TBD, belief_TB1], dim=-1)             # [T,B,D+1]
        return self.fe.forward_seq(x, h0)                     # feat_TBH, hT

    def encode_target_seq(self, obs_TBD: th.Tensor, belief_TB1: th.Tensor, h0: th.Tensor):
        x = th.cat([obs_TBD, belief_TB1], dim=-1)
        return self.fe_t.forward_seq(x, h0)

    def sample_from_features(self, feat: th.Tensor, detach_feat_for_actor: bool = True):
        x = feat.detach() if detach_feat_for_actor else feat
        # Avoid strict validator (and small runtime cost) since we already clamped
        mu, std = self._mean_std(x)
        dist = th.distributions.Normal(mu, std, validate_args=False)
        u = dist.rsample()
        a = th.tanh(u)
        logp_n = dist.log_prob(u).sum(-1, True) - self._tanh_log_det(u).sum(-1, True)
        if self.scale_log_offset:
            logp_n = logp_n - self.scale_log_offset
        return a, logp_n

    def q(self, feat: th.Tensor, a: th.Tensor):
        x = th.cat([feat, a], dim=-1)
        return self.q1(x), self.q2(x)

    @th.no_grad()
    def target_backup_seq(self, nxt_TBD: th.Tensor, belief_TB1: th.Tensor, h_seq_next: th.Tensor,
                        r_TB1: th.Tensor, d_TB1: th.Tensor, gamma: float, alpha: th.Tensor):
        # 1) Do NOT reuse online hidden. Start from target's clean zero hidden.
        B = nxt_TBD.size(1)
        h0_tgt = self.fe_t.init_state(B, nxt_TBD.device)

        # 2) Encode the next sequence with target encoder from zero state
        feat_n_TBH, _ = self.encode_target_seq(nxt_TBD, belief_TB1, h_seq_next)   # [T,B,H]

        # 4) Actor heads in fp32 (your _mean_std already does fp32 math + clamps)
        mu, std = self._mean_std(feat_n_TBH)

        # 5) Build the distribution without extra validation (we clamp already)
        dist = th.distributions.Normal(mu, std, validate_args=False)
        u = dist.rsample()
        a_n = th.tanh(u)

        logp_n = dist.log_prob(u).sum(-1, True) - self._tanh_log_det(u).sum(-1, True)
        if self.scale_log_offset:
            logp_n = logp_n - self.scale_log_offset

        qmin = th.min(self.q1_t(th.cat([feat_n_TBH, a_n], -1)),
                    self.q2_t(th.cat([feat_n_TBH, a_n], -1)))
        return r_TB1 + (1 - d_TB1) * gamma * (qmin - alpha * logp_n)
    
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
