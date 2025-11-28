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
        self.mlp  = nn.Sequential(nn.Linear(obs_dim, hidden), nn.GELU())
        self.gru  = nn.GRU(input_size=hidden, hidden_size=hidden, num_layers=1, batch_first=False)

    def init_state(self, bsz: int, device=None):
        return th.zeros(1, bsz, self.hidden, device=device)
    
    def encode(self, obs: th.Tensor, h0: th.Tensor = None) -> th.Tensor:
        if obs.dim() == 2:
            B, D = obs.shape
            h0 = self.init_state(B, device=obs.device) if h0 is None else h0
            x = self.mlp(obs)                             # [B,H]
            x = x.unsqueeze(0)                            # [1,B,H] as time=1
            y, h_next = self.gru(x, h0)                       # y:[1,B,H], h1:[1,B,H]
        elif obs.dim() == 3:
            T, B, D = obs.shape
            h0 = self.init_state(B, device=obs.device) if h0 is None else h0
            x = self.mlp(obs)                               # [T,B,H]
            y, h_next = self.gru(x, h0)                             # y:[T,B,H]
        else:
            raise ValueError("obs must be 2D or 3D tensor")
        return y, h_next


class Network(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=128, belief_dim=1,
                 init_log_std=-2.0, log_std_min=-20.0, log_std_max=2.0, scale_log_offset=None):
        super().__init__()
        self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)
        # Critics
        def _make_q():
            return nn.Sequential(
                nn.Linear(hidden + act_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden),          nn.GELU(),
                nn.Linear(hidden, 1),
            )

        # Belief
        self.belief_encoder_gru = FeatureExtractorGRU(obs_dim, hidden)
        self.belief_encoder_mu  = nn.Linear(hidden, belief_dim)
        self.belief_encoder_var = nn.Linear(hidden, belief_dim)
        
        self.belief_decoder     = nn.Sequential(
            nn.Linear(belief_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

        self.q1, self.q2        = _make_q(), _make_q()
        
        self.fe_t               = FeatureExtractorGRU(obs_dim + belief_dim, hidden) 
        self.q1_t, self.q2_t    = _make_q(), _make_q()

        # Actor
        self.fe                 = FeatureExtractorGRU(obs_dim + belief_dim, hidden)
        self.mu                 = nn.Linear(hidden, act_dim)
        self.logstd_head        = nn.Linear(hidden, act_dim)
        # self.logstd_head        = nn.Parameter(th.full((act_dim,), init_log_std))

        self._init_weights(init_log_std)
        
        self._hard_sync()
        self.scale_log_offset = scale_log_offset

    def _init_weights(self, init_log_std: float):
        gain = nn.init.calculate_gain("relu")  # good for GELU/ReLU-like nets

        def _orthogonal_init(m: nn.Module):
            # Linear layers
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            # GRU: orthogonal for both input->hidden and hidden->hidden weights
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name or "weight_hh" in name:
                        nn.init.orthogonal_(param.data, gain=gain)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

        # Apply to all submodules (FeatureExtractorGRU included)
        self.apply(_orthogonal_init)

        # Special: initialize log-std head bias to given value
        if self.logstd_head.bias is not None:
            self.logstd_head.bias.data.fill_(init_log_std)

    def _hard_sync(self):
        for t, s in zip(self.q1_t.parameters(), self.q1.parameters()): t.data.copy_(s.data)
        for t, s in zip(self.q2_t.parameters(), self.q2.parameters()): t.data.copy_(s.data)
        for t, s in zip(self.fe_t.parameters(), self.fe.parameters()): t.data.copy_(s.data)
        
    def soft_sync(self, tau):
        with th.no_grad():
            for tp, sp in zip(self.q1_t.parameters(), self.q1.parameters()):
                tp.mul_(1 - tau).add_(sp, alpha=tau)
            for tp, sp in zip(self.q2_t.parameters(), self.q2.parameters()):
                tp.mul_(1 - tau).add_(sp, alpha=tau)
            for tp, sp in zip(self.fe_t.parameters(), self.fe.parameters()):
                tp.mul_(1 - tau).add_(sp, alpha=tau)
                
    @staticmethod
    def _tanh_log_det(u: th.Tensor) -> th.Tensor:
        # return 2 * (th.log(th.tensor(2.0, device=u.device)) - u - F.softplus(-2*u))
        return th.log(1.0 - th.tanh(u) ** 2 + 1e-6)

    def belief_encode(self, obs_BD: th.Tensor, b_h: th.Tensor = None, is_evaluate = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        feat_BH, b_h_next   = self.belief_encoder_gru.encode(obs_BD, b_h)
        mu_BH               = self.belief_encoder_mu(feat_BH)
        logvar_BH           = self.belief_encoder_var(feat_BH)

        # reparameterize
        if is_evaluate:
            z_BH = th.distributions.Normal(mu_BH, (logvar_BH / 2).exp()).mode
        else:
            z_BH = th.distributions.Normal(mu_BH, (logvar_BH / 2).exp()).rsample()
        return z_BH, b_h_next, mu_BH, logvar_BH
    
    def belief_decode(self, z_BH: th.Tensor) -> th.Tensor:
        return self.belief_decoder(z_BH)                 # [B, 1]

    def feature_compute(self, obs_BD: th.Tensor, z_BH: th.Tensor, f_h: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor]:
        return self.fe.encode(th.cat([obs_BD, z_BH], dim=-1), f_h)   # [T,B,H]

    def action_compute(self, feat: th.Tensor, is_evaluate = False):
        # penultimate normalization (pnorm)
        feat = feat / feat.norm(p=1, dim=-1, keepdim=True).clamp_min(1e-8)
        mean_action = self.mu(feat)
        # std = th.clamp(self.logstd_head, self.log_std_min, self.log_std_max).exp()
        std = th.clamp(self.logstd_head(feat), self.log_std_min, self.log_std_max).exp()
        
        dist = th.distributions.Normal(mean_action, std)
        
        if is_evaluate:
            u = dist.mode
        else:
            u = dist.rsample()
        
        normal_log = dist.log_prob(u).sum(-1, True) 
        logp_n = normal_log - self._tanh_log_det(u).sum(-1, True)
        
        a = th.tanh(u)
        return a, logp_n

    def critic_compute(self, feat: th.Tensor, a: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        x = th.cat([feat, a], dim=-1)
        return self.q1(x), self.q2(x)

    @th.no_grad()
    def target_backup_seq(
        self, 
        nxt_TBD: th.Tensor, 
        r_TB1: th.Tensor, 
        d_TB1: th.Tensor, 
        gamma: float, 
        alpha: th.Tensor,
        b_h: Optional[th.Tensor] = None,
        f_h: Optional[th.Tensor] = None
    ) ->  Tuple[th.Tensor, th.Tensor]:
        # 2) Encode the next sequence with target encoder from zero state
        z_TBH, _, _, _ = self.belief_encode(nxt_TBD, b_h, is_evaluate=True)
        
        feat_TBH, _ = self.fe_t.encode(th.cat([nxt_TBD, z_TBH], dim=-1), f_h)   # [T,B,H]

        a_TBA, logp_TB1 = self.action_compute(feat_TBH, is_evaluate=False)  # [T,B,A], [T,B,1]

        qmin = th.min(
            self.q1_t(th.cat([feat_TBH, a_TBA], dim=-1)),
            self.q2_t(th.cat([feat_TBH, a_TBA], dim=-1))
        )
        
        return r_TB1 + (1 - d_TB1) * gamma * (qmin - alpha * logp_TB1), qmin - alpha * logp_TB1


    # param groups
    def actor_parameters(self):
        return list(self.fe.parameters()) + list(self.mu.parameters()) + list(self.logstd_head.parameters())
    
    def critic_parameters(self):
        return list(self.q1.parameters()) + list(self.q2.parameters())
    
    def belief_parameters(self):
        return list(self.belief_encoder_gru.parameters()) + list(self.belief_encoder_mu.parameters()) + list(self.belief_encoder_var.parameters()) + list(self.belief_decoder.parameters())
