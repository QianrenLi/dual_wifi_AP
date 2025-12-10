from typing import Optional, Tuple
import contextlib
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from util.reward_to_value_bound import ValueDistribution


class FeatureExtractorGRU(nn.Module):
    def __init__(self, obs_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.mlp = nn.Sequential(nn.Linear(obs_dim, hidden), nn.GELU())
        self.gru = nn.GRU(input_size=hidden, hidden_size=hidden, num_layers=1, batch_first=False)

    def init_state(self, bsz: int, device=None):
        return th.zeros(1, bsz, self.hidden, device=device)

    def _encode(self, obs: th.Tensor, h0: th.Tensor = None) -> Tuple[th.Tensor, th.Tensor]:
        if obs.dim() == 2:
            B, D = obs.shape
            h0 = self.init_state(B, device=obs.device) if h0 is None else h0
            x = self.mlp(obs)
            x = x.unsqueeze(0)
            y, h_next = self.gru(x, h0)
        elif obs.dim() == 3:
            T, B, D = obs.shape
            h0 = self.init_state(B, device=obs.device) if h0 is None else h0
            x = self.mlp(obs)
            y, h_next = self.gru(x, h0)
        else:
            raise ValueError("obs must be 2D or 3D tensor")
        return y, h_next


class Network(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        bins: int,
        hidden: int = 128,
        belief_dim: int = 1,
        belief_labels_dim: int = 5,
        init_log_std: float = -2.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        n_critics: int = 2,
        dropped_per_critic: int = 5,
    ):
        super().__init__()
        self.log_std_min, self.log_std_max = float(log_std_min), float(log_std_max)
        self.n_quantiles = bins
        self.n_critics = n_critics
        self.dropped_per_critic = dropped_per_critic

        def _make_q():
            return nn.Sequential(
                nn.Linear(hidden + belief_dim + act_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, self.n_quantiles),
            )

        self.belief_encoder_gru = FeatureExtractorGRU(obs_dim, hidden)
        self.belief_encoder_mu = nn.Sequential(
            nn.Linear(hidden, belief_dim),
            nn.Tanh()
        )
        self.belief_encoder_log_std = nn.Sequential(
            nn.Linear(hidden, belief_dim),
        )
        self.belief_decoder = nn.Sequential(
            nn.Linear(belief_dim, belief_labels_dim),
        )

        self.critic_fe = FeatureExtractorGRU(obs_dim, hidden)
        self.critic_fe_t = FeatureExtractorGRU(obs_dim, hidden)
        self.critics = nn.ModuleList([_make_q() for _ in range(self.n_critics)])
        self.critics_t = nn.ModuleList([_make_q() for _ in range(self.n_critics)])

        self.actor_fe = FeatureExtractorGRU(obs_dim, hidden)
        self.actor_l1 = nn.Sequential(
            nn.Linear(hidden + belief_dim, hidden),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.logstd_head = nn.Linear(hidden, act_dim)

        self._init_weights(init_log_std)
        self._hard_sync()

    def _init_weights(self, init_log_std: float):
        gain = nn.init.calculate_gain("relu")

        def _orthogonal_init(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name or "weight_hh" in name:
                        nn.init.orthogonal_(param.data, gain=gain)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

        self.apply(_orthogonal_init)
        if self.logstd_head.bias is not None:
            self.logstd_head.bias.data.fill_(init_log_std)

    def _hard_sync(self):
        for ct, c in zip(self.critics_t, self.critics):
            for t_p, s_p in zip(ct.parameters(), c.parameters()):
                t_p.data.copy_(s_p.data)
        for t, s in zip(self.critic_fe_t.parameters(), self.critic_fe.parameters()):
            t.data.copy_(s.data)

    def soft_sync(self, tau: float):
        with th.no_grad():
            for ct, c in zip(self.critics_t, self.critics):
                for tp, sp in zip(ct.parameters(), c.parameters()):
                    tp.mul_(1 - tau).add_(sp, alpha=tau)
            for tp, sp in zip(self.critic_fe_t.parameters(), self.critic_fe.parameters()):
                tp.mul_(1 - tau).add_(sp, alpha=tau)

    @staticmethod
    def _tanh_log_det(u: th.Tensor) -> th.Tensor:
        return 2 * (th.log(th.tensor(2.0, device=u.device)) - u - F.softplus(-2 * u))

    def belief_encode(
        self,
        obs: th.Tensor,
        b_h: th.Tensor = None,
        is_evaluate: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        feat, b_h_next = self.belief_encoder_gru._encode(obs, b_h)
        mu_v = self.belief_encoder_mu(feat)
        logstd = self.belief_encoder_log_std(feat)
        std = 1e-3 + (1 - 1e-3) * F.sigmoid(logstd)
        dist = th.distributions.Normal(mu_v, std)
        if is_evaluate:
            latent = dist.mode
        else:
            latent = dist.rsample()
        return latent, b_h_next, mu_v, th.log(std) * 2

    def belief_decode(self, latent: th.Tensor) -> th.Tensor:
        return self.belief_decoder(latent)

    def actor_feature_compute(
        self,
        obs: th.Tensor,
        latent: th.Tensor,
        f_h: th.Tensor = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        feat, f_h_n = self.actor_fe._encode(obs, f_h)
        return th.cat([feat, latent], dim=-1), f_h_n

    def critic_feature_compute(
        self,
        obs: th.Tensor,
        latent: th.Tensor,
        f_h: th.Tensor = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        feat, f_h_n = self.critic_fe._encode(obs, f_h)
        return th.cat([feat, latent], dim=-1), f_h_n

    def action_compute(
        self,
        feat: th.Tensor,
        is_evaluate: bool = False,
        return_stats: bool = False
    ):
        feat = self.actor_l1(feat)
        mean_action = self.mu(feat)
        logstd = self.logstd_head(feat)
        std = 1e-3 + (1 - 1e-3) * F.sigmoid(logstd)
        dist = th.distributions.Normal(mean_action, std)
        if is_evaluate:
            u = dist.mode
        else:
            u = dist.rsample()
        normal_log = dist.log_prob(u).sum(-1, True)
        logp_n = normal_log - self._tanh_log_det(u).sum(-1, True)
        a = th.tanh(u)
        if return_stats:
            return a, logp_n, mean_action, std
        return a, logp_n

    def critic_compute(self, feat: th.Tensor, a: th.Tensor) -> th.Tensor:
        x = th.cat([feat, a], dim=-1)
        zs = []
        for q in self.critics:
            zs.append(q(x).unsqueeze(-2))
        return th.cat(zs, dim=-2)

    @th.no_grad()
    def target_backup_seq(
        self,
        nxt_TBD: th.Tensor,
        r_TB1: th.Tensor,
        d_TB1: th.Tensor,
        gamma: float,
        alpha: th.Tensor,
        b_h: Optional[th.Tensor] = None,
        f_h_critic: Optional[th.Tensor] = None,
        f_h_actor: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        z_TBH, _, _, _ = self.belief_encode(nxt_TBD, b_h, is_evaluate=True)

        # feat_actor_TBH, _ = self.actor_fe._encode(nxt_TBD, f_h_actor)
        # feat_actor_TBH = th.cat([feat_actor_TBH, z_TBH], dim=-1)
        feat_actor_TBH, _ = self.actor_feature_compute(nxt_TBD, z_TBH, f_h_actor)
        a_TBA, logp_TB1 = self.action_compute(feat_actor_TBH, is_evaluate=False)

        feat_critic_TBH, _ = self.critic_fe_t._encode(nxt_TBD, f_h_critic)
        feat_critic_TBH = th.cat([feat_critic_TBH, z_TBH], dim=-1)

        xa = th.cat([feat_critic_TBH, a_TBA], dim=-1)
        z_list = []
        for qt in self.critics_t:
            z_list.append(qt(xa).unsqueeze(2))
        z_cat = th.cat(z_list, dim=-2)

        T, B, C, N = z_cat.shape
        z_flat = z_cat.reshape(T, B, C * N)
        z_sorted, _ = th.sort(z_flat, dim=-1)

        total_atoms = C * N
        total_dropped = self.dropped_per_critic * C
        total_kept = total_atoms - total_dropped
        z_trunc = z_sorted[..., :total_kept]

        target_atoms = r_TB1 + gamma * (1.0 - d_TB1) * (z_trunc - alpha * logp_TB1)
        return target_atoms

    @contextlib.contextmanager
    def critics_frozen(self):
        ps = self.critic_parameters()
        flags = [p.requires_grad for p in ps]
        for p in ps:
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, f in zip(ps, flags):
                p.requires_grad_(f)

    def actor_parameters(self):
        return (
            list(self.actor_fe.parameters())
            + list(self.actor_l1.parameters())
            + list(self.mu.parameters())
            + list(self.logstd_head.parameters())
        )

    def critic_parameters(self):
        crit_params = []
        for q in self.critics:
            crit_params += list(q.parameters())
        crit_params += list(self.critic_fe.parameters())
        return crit_params

    def belief_parameters(self):
        return (
            list(self.belief_encoder_gru.parameters())
            + list(self.belief_encoder_mu.parameters())
            + list(self.belief_encoder_log_std.parameters())
            + list(self.belief_decoder.parameters())
        )
