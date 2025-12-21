from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch as th
import torch.nn.functional as F
import importlib

from net_util.base import PolicyBase
from net_util.replay.base import BaseReplayBuffer
from net_util.model.ablation_belief import Network
from util.reward_to_value_bound import ValueDistribution
from .. import register_policy, register_policy_cfg
from torch.utils.tensorboard import SummaryWriter

Tensor = th.Tensor

def symlog(x: th.Tensor, eps=1e-12) -> th.Tensor:
    return th.sign(x) * th.log(th.abs(x) + 1.0 + eps)

def _flatten_params(m: Optional[th.nn.Module] | list) -> th.Tensor:
    if m is None:
        return th.zeros(0)
    if isinstance(m, list):
        parts = [p.detach().view(-1).cpu() for p in m if p.requires_grad]
        return th.cat(parts) if parts else th.zeros(0)
    parts = [p.detach().view(-1).cpu() for p in m.parameters() if p.requires_grad]
    return th.cat(parts) if parts else th.zeros(0)

def quantile_huber_loss(pred: Tensor, target: Tensor, taus: Tensor, kappa: float = 1.0) -> Tensor:
    """
    QR loss as in Dabney et al.
    pred:   [..., M]   (critic quantiles θ^m)
    target: [..., K]   (target atoms y_i)
    taus:   [M]        (quantile fractions τ_m)
    returns: [...], loss per sample (averaged over m,i)
    """
    # [..., M, K]
    # print(target.unsqueeze(-2).shape)
    # print(pred.unsqueeze(-1).shape)
    td = target.unsqueeze(-2).unsqueeze(-3) - pred.unsqueeze(-1)
    abs_td = td.abs()
    huber = th.where(
        abs_td <= kappa,
        0.5 * td.pow(2),
        kappa * (abs_td - 0.5 * kappa),
    )

    # broadcast τ_m over the M dimension
    taus = taus.view(1, 1, 1, -1, 1).to(pred.device)
    loss = th.abs(taus - (td.detach() < 0).float()) * huber  # [T,B,C,N,K]

    # average over quantile index m (N) and sample index i (K)
    return loss.mean(dim=(-1, -2))  # [T,B,C]


@register_policy_cfg
@dataclass
class ABLATIONBELIEF_Config:
    batch_size: int = 256
    learning_starts: int = 2_000
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    target_update_interval: int = 1
    ent_coef: str | float = "auto"
    ent_coef_init: float = 1.0
    target_entropy: str | float = "auto"
    act_dim: int = 2
    obs_dim: int = 6
    device: str = "cpu"
    seed: int = 0
    network_module_path: str = "net_util.model.ablation_belief"
    load_path: str = "latest.pt"
    batch_rl: bool = False
    retrain_interval: int = 0
    retrain_reset_alpha: bool = True
    retrain_reset_targets: bool = True
    critic_utd: int = 1
    max_utd_ratio: float = 1.0
    log_interval: int = 50
    cdl_num_random: int = 10
    cdl_beta_cql_multiplier: float = 5 * 20
    annealing_max_lb: float = 0.05
    annealing_epoch_max: float = 10
    sigma_k = 0.1

    MAX_BITRATE = 3e7
    MAX_OUTAGE = 1.0


@register_policy
class ABLATIONBELIEF(PolicyBase):
    def __init__(
        self,
        cmd_cls: Any,
        cfg: ABLATIONBELIEF_Config,
        rollout_buffer: BaseReplayBuffer | None = None,
        device: str | None = None,
        state_transform_dict: dict | None = None,
        reward_cfg: Optional[dict] = None,
    ):
        super().__init__(cmd_cls, state_transform_dict=state_transform_dict, reward_cfg=reward_cfg)
        
        self.cfg = cfg
        self.device = th.device(device or cfg.device)
        th.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        th.backends.cudnn.benchmark = True
        
        bins = 26

        try:
            net_mod = importlib.import_module(cfg.network_module_path)
            net_class = getattr(net_mod, "Network")
        except (ImportError, AttributeError) as e:
            raise RuntimeError(
                f"Failed to import Network class from {cfg.network_module_path}. "
                f"Error: {e}. Please check the module path and ensure the Network class exists."
            ) from e

        self.net: Network = net_class(cfg.obs_dim, cfg.act_dim, bins=bins, n_critics = 5).to(self.device)  # type: ignore

        self.buf = rollout_buffer

        ac_lr = cfg.lr * (0.1 if cfg.batch_rl else 1.0)
        self.actor_opt = th.optim.Adam(self.net.actor_parameters(), lr=ac_lr)
        self.critic_opt = th.optim.Adam(self.net.critic_parameters(), lr=cfg.lr)
        
        self.log_alpha, self.alpha_opt, self.alpha_tensor = None, None, None
        self.target_entropy: float | None = None
        if cfg.ent_coef == "auto" and not cfg.batch_rl:
            self.log_alpha = th.tensor([cfg.ent_coef_init], device=self.device).log().requires_grad_(True)
            self.alpha_opt = th.optim.Adam([self.log_alpha], lr=cfg.lr)
            self.target_entropy = -float(cfg.act_dim) if cfg.target_entropy == "auto" else float(cfg.target_entropy)
        else:
            alpha_val = float(cfg.ent_coef) if isinstance(cfg.ent_coef, float) else 0.2
            self.alpha_tensor = th.tensor(alpha_val, device=self.device)

        self._upd = 0
        self._global_step = 0
        self._eval_h: Tensor | None = None

        self.actor_loss_scale_S: th.Tensor = th.tensor(1.0, device=self.device)


    def _alpha(self) -> Tensor:
        if self.log_alpha is not None:
            alpha = self.log_alpha.exp()
            return alpha.detach()
        else:
            return self.alpha_tensor.detach()

    def _get_beta(self, epoch: int) -> float:
        warmup_steps = self.cfg.annealing_epoch_max 
        max_beta = self.cfg.annealing_max_lb
        
        if epoch >= warmup_steps:
            return max_beta
        
        return max(max_beta * (epoch / warmup_steps), 0)
    
    def update_actor_loss_scale(self, returns_train: th.Tensor,
                                alpha: float = 0.99) -> th.Tensor:
        """
        returns_train: [T, B, 1] lambda-returns
        Updates self.actor_loss_scale_S (scalar EMA).
        """
        with th.no_grad():
            r = returns_train.squeeze(-1)  # [T, B]

            # compute 5th and 95th percentile over batch dim at once
            q = th.tensor([0.05, 0.95], device=r.device, dtype=r.dtype)
            per = th.quantile(r, q, dim=1)    # shape [2, T]
            delta_all = per[1] - per[0]       # [T]

            S = self.actor_loss_scale_S
            for delta in delta_all:           # EMA over time
                S = alpha * S + (1.0 - alpha) * delta

            self.actor_loss_scale_S = S

        return self.actor_loss_scale_S

    @staticmethod
    def _step_with_clip(params_iterable, opt: th.optim.Optimizer, loss: Tensor, clip_norm: float | None) -> None:
        opt.zero_grad()
        loss.backward()
        if clip_norm is not None:
            th.nn.utils.clip_grad_norm_(params_iterable, clip_norm)
        opt.step()

    def _log_batch_stats(self, writer: SummaryWriter, step: int, **kwargs) -> None:
        writer.add_scalar("loss/actor", kwargs["a_loss"].item(), step)
        writer.add_scalar("loss/critic", kwargs["c_loss"].item(), step)

        if self.alpha_opt is not None and kwargs.get("ent_loss") is not None:
            writer.add_scalar("loss/entropy", kwargs["ent_loss"].item(), step)
            writer.add_scalar("policy/alpha", self._alpha().item(), step)

        writer.add_scalar("q/qmin_pi", kwargs["qmin_pi"].mean().item(), step)
        writer.add_scalar("policy/logp_pi", kwargs["logp_TB1"].mean().item(), step)

        if "qdist_avg" in kwargs:
            writer.add_histogram("q/dist_qmin_pi", kwargs["qdist_avg"], step)

        if "mu_mean" in kwargs:
            mu_mean = kwargs["mu_mean"]
            mu_std = kwargs["mu_std"]
            sigma_mean = kwargs["sigma_mean"]
            sigma_std = kwargs["sigma_std"]
            act_mean = kwargs["act_mean"]
            act_std = kwargs["act_std"]
            for i in range(mu_mean.shape[0]):
                writer.add_scalar(f"policy/mu_mean_dim{i}", mu_mean[i].item(), step)
                writer.add_scalar(f"policy/mu_std_dim{i}", mu_std[i].item(), step)
                writer.add_scalar(f"policy/sigma_mean_dim{i}", sigma_mean[i].item(), step)
                writer.add_scalar(f"policy/sigma_std_dim{i}", sigma_std[i].item(), step)
                writer.add_scalar(f"policy/action_mean_dim{i}", act_mean[i].item(), step)
                writer.add_scalar(f"policy/action_std_dim{i}", act_std[i].item(), step)

        if "q_region_stats" in kwargs:
            q_region_stats = kwargs["q_region_stats"]
            for region_val, q_mean_r in q_region_stats.items():
                writer.add_scalar(
                    f"region{region_val}/qmin_pi_mean",
                    q_mean_r.item(),
                    step,
                )

        if "act_region_stats" in kwargs:
            act_region_stats = kwargs["act_region_stats"]
            for region_val, act_mean_r in act_region_stats.items():
                for i in range(act_mean_r.shape[0]):
                    writer.add_scalar(
                        f"region{region_val}/action_mean_dim{i}",
                        act_mean_r[i].item(),
                        step,
                    )

        if "rew_region_stats" in kwargs:
            rew_region_stats = kwargs["rew_region_stats"]
            for region_val, rew_mean_r in rew_region_stats.items():
                writer.add_scalar(
                    f"region{region_val}/reward_mean",
                    rew_mean_r.item(),
                    step,
                )

        if "critic_lr" in kwargs:
            writer.add_scalar("opt/critic_lr", kwargs["critic_lr"], step)

        if "reward_mean" in kwargs:
            writer.add_scalar("data/reward_mean", kwargs["reward_mean"], step)
        if "reward_std" in kwargs:
            writer.add_scalar("data/reward_std", kwargs["reward_std"], step)

        if "interference" in kwargs:
            writer.add_histogram("data/interference", kwargs["interference"], step)

        if "is_weights" in kwargs:
            writer.add_histogram("data/is_weights", kwargs["is_weights"], step)

        if "actor_param_norm" in kwargs:
            writer.add_scalar("param_norm/actor", kwargs["actor_param_norm"], step)
        if "critic_param_norm" in kwargs:
            writer.add_scalar("param_norm/critic", kwargs["critic_param_norm"], step)

        dbg = getattr(self, "_last_debug", None)
        if dbg is not None and self._global_step % self.cfg.log_interval == 0:
            writer.add_scalar("debug/q_mean", dbg["q_mean"], self._global_step)
            writer.add_scalar("debug/backup_mean", dbg["backup_mean"], self._global_step)
            writer.add_scalar("debug/dq_backup_minus_q", dbg["dq"], self._global_step)

        bdbg = getattr(self.net, "_backup_debug", None)
        if bdbg is not None and self._global_step % self.cfg.log_interval == 0:
            writer.add_scalar("backup/r_mean", bdbg["r_mean"], self._global_step)
            writer.add_scalar("backup/v_part_mean", bdbg["v_part_mean"], self._global_step)
            writer.add_scalar("backup/ent_part_mean", bdbg["ent_part_mean"], self._global_step)
            writer.add_scalar("backup/z_trunc_mean", bdbg["z_trunc_mean"], self._global_step)
            
        actordbg = getattr(self, "_actor_debug", None)
        if actordbg is not None and self._global_step % self.cfg.log_interval == 0:
            writer.add_scalar("actor_debug/weight_logp", actordbg["weight_logp"], self._global_step)
            writer.add_scalar("actor_debug/weight_q", actordbg["weight_q"], self._global_step)


    
    def _critic_loss(
        self,
        obs_TBD: Tensor,
        act_TBA: Tensor,
        nxt_TBD: Tensor,
        rew_TB1: Tensor,
        done_TB1: Tensor,
        importance_weights: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:

        backup_q_TB1 = self.net.target_backup_seq(
            nxt_TBD,
            rew_TB1,
            done_TB1,
            self.cfg.gamma,
            self._alpha(),
        )                        # [T,B,K_total_kept]

        z_TBCN = self.net.critic_compute(obs_TBD, act_TBA)  # [T,B,C,N]
        _, _, _, N = z_TBCN.shape
        
        taus = (2 * th.arange(1, N+1, device=z_TBCN.device) - 1) / (2.0 * N)

        loss_TBC = quantile_huber_loss(z_TBCN, backup_q_TB1, taus)   # [T,B,C]
        c_loss_batch = loss_TBC.mean(dim=-1).mean(dim=0)             # [B]

        c_loss = (c_loss_batch * importance_weights).sum() / (importance_weights.sum() + 1e-8)

        return c_loss, c_loss_batch.detach(), {}
    

    def _actor_loss(
        self,
        obs_TBD: Tensor,
        a_pi_TBA: Tensor,
        logp_TB1: Tensor,
        returns_train: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        alpha = self._alpha()
        z_TBCN = self.net.critic_compute(obs_TBD, a_pi_TBA)   # [T,B,C,N]

        q_TBC = z_TBCN.mean(dim=-1)                            # [T,B,C]
        qmin_pi = q_TBC.mean(dim=-1, keepdim=True)             # [T,B,1]

        scale_value = self.update_actor_loss_scale(returns_train.detach()).item() / 3.3
        denom = max(scale_value, 1.0)

        # two parts of the loss
        loss_logp_TB1 = alpha * logp_TB1
        loss_q_TB1 = -(qmin_pi - returns_train.detach()) / denom

        # total actor loss
        a_loss = (loss_logp_TB1 + loss_q_TB1).mean()

        qdist_avg = z_TBCN.mean(dim=(0, 1, 2)).detach()        # [N]

        return a_loss, qmin_pi.detach(), qdist_avg

    def _alpha_loss(self, logp_TB1: Tensor) -> Tensor:
        if self.alpha_opt is None or self.log_alpha is None:
            return th.zeros((), device=self.device)
        return -(self.log_alpha * (logp_TB1.detach() + self.target_entropy)).mean()  # type: ignore

            

    def train_per_epoch(
        self,
        epoch: int,
        writer: Optional[SummaryWriter] = None,
        is_batch_rl: bool | None = None,
        log_dir: str = "runs/sac_rnn",
    ) -> bool:
        is_batch_rl = is_batch_rl if is_batch_rl is not None else self.cfg.batch_rl

        local_writer = writer is None
        writer = writer or SummaryWriter(log_dir=log_dir)

        ent_loss: Optional[Tensor] = None

        burn_in = 50

        obs_TBD, act_TBA, rew_TB1, nxt_TBD, done_TB1, info = next(
            self.buf.get_sequences(self.cfg.batch_size, trace_length=100)
        )

        obs_burn = obs_TBD[:burn_in]
        obs_train = obs_TBD[burn_in:]
        act_train = act_TBA[burn_in:]
        rew_train = rew_TB1[burn_in:]
        nxt_train = nxt_TBD[burn_in:]
        done_train = done_TB1[burn_in:]
        returns_train = info["returns"][burn_in:]

        importance_weights: Tensor = info["is_weights"].squeeze(-1)
        interference: Tensor = info["interference"].squeeze(-1)

        # Actor takes obs directly (no belief states)
        a_pi_TBA, logp_TB1, mu_TBA, std_TBA = self.net.action_compute(
            obs_train, return_stats=True
        )
        
        if self.alpha_opt is not None:
            ent_loss = self._alpha_loss(logp_TB1.detach())
            self._step_with_clip([self.log_alpha], self.alpha_opt, ent_loss, clip_norm=5.0)  # type: ignore
        
        c_loss, c_loss_batch, _ = self._critic_loss(
            obs_train,
            act_train,
            nxt_train,
            rew_train,
            done_train,
            importance_weights,
        )
        self._step_with_clip(self.net.critic_parameters(), self.critic_opt, c_loss, clip_norm=10.0)
        self.net.soft_sync(self.cfg.tau)

        with self.net.critics_frozen():
            a_loss, qmin_pi, qdist_avg = self._actor_loss(
                obs_train, a_pi_TBA, logp_TB1, returns_train=returns_train
            )
            self._step_with_clip(self.net.actor_parameters(), self.actor_opt, a_loss, clip_norm=5.0)

            mu_mean = mu_TBA.mean(dim=(0, 1))
            mu_std = mu_TBA.std(dim=(0, 1))
            sigma_mean = std_TBA.mean(dim=(0, 1))
            sigma_std = std_TBA.std(dim=(0, 1))
            act_mean = a_pi_TBA.mean(dim=(0, 1))
            act_std = a_pi_TBA.std(dim=(0, 1))

            mu_region_stats: Dict[int, Tuple[Tensor, Tensor]] = {}
            q_region_stats: Dict[int, Tensor] = {}
            act_region_stats: Dict[int, Tensor] = {}
            rew_region_stats: Dict[int, Tensor] = {}

            unique_regions = interference.unique()
            q_B = qmin_pi.mean(dim=0).squeeze(-1)
            rew_B = rew_train.mean(dim=0).squeeze(-1)
            act_BA = a_pi_TBA.mean(dim=0)

            for r in unique_regions:
                mask_B = interference == r
                if mask_B.any():
                    mu_region = mu_TBA[:, mask_B, :]
                    mu_mean_r = mu_region.mean(dim=(0, 1))
                    mu_std_r = mu_region.std(dim=(0, 1))
                    mu_region_stats[int(r.item())] = (mu_mean_r, mu_std_r)

                    q_region_stats[int(r.item())] = q_B[mask_B].mean()
                    rew_region_stats[int(r.item())] = rew_B[mask_B].mean()
                    act_region_stats[int(r.item())] = act_BA[mask_B].mean(dim=0)

        if self._global_step % self.cfg.log_interval == 0:
                reward_mean = rew_train.mean().item()
                reward_std = rew_train.std().item()
                actor_flat = _flatten_params(list(self.net.actor_parameters()))
                critic_flat = _flatten_params(list(self.net.critic_parameters()))

                self._log_batch_stats(
                    writer,
                    self._global_step,
                    a_loss=a_loss,
                    c_loss=c_loss,
                    ent_loss=ent_loss,
                    logp_TB1=logp_TB1,
                    qmin_pi=qmin_pi,
                    is_batch_rl=is_batch_rl,
                    qdist_avg=qdist_avg,
                    mu_mean=mu_mean,
                    mu_std=mu_std,
                    sigma_mean=sigma_mean,
                    sigma_std=sigma_std,
                    act_mean=act_mean,
                    act_std=act_std,
                    mu_region_stats=mu_region_stats,
                    q_region_stats=q_region_stats,
                    act_region_stats=act_region_stats,
                    rew_region_stats=rew_region_stats,
                    reward_mean=reward_mean,
                    reward_std=reward_std,
                    interference=interference.detach().cpu().float(),
                    is_weights=importance_weights.detach().cpu().view(-1),
                    actor_param_norm=actor_flat.norm().item() if actor_flat.numel() > 0 else 0.0,
                    critic_param_norm=critic_flat.norm().item() if critic_flat.numel() > 0 else 0.0,
                )

                writer.add_histogram(
                    "data/reward_hist",
                    rew_train.detach().cpu().view(-1),
                    self._global_step,
                )

        self._global_step += 1
        self.buf.update_episode_losses(info["ep_ids"], c_loss_batch.cpu().numpy())  # type: ignore

        if local_writer:
            writer.flush()
            writer.close()

        if self._global_step % 250 == 0:
            return False
            
        return True

    @th.no_grad()
    def tf_act(
        self,
        obs_vec: list[float],
        is_evaluate: bool = False,
        reset_hidden: bool = False,
    ) -> Dict[str, np.ndarray]:
        obs = th.tensor(
            obs_vec, device=self.device, dtype=th.float32
        ).unsqueeze(0).unsqueeze(0)

        # Actor takes obs directly (no belief states)
        action, logp = self.net.action_compute(obs, is_evaluate)
        v_like = 0

        return {
            "action": action[0][0].cpu().numpy(),
            "log_prob": logp[0][0].cpu().numpy(),
            "value": v_like,
        }

    def save(self, path: str):
        checkpoint = {
            "model_state_dict": self.net.state_dict(),
            "actor_opt_state_dict": self.actor_opt.state_dict(),
            "critic_opt_state_dict": self.critic_opt.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_opt_state_dict": self.alpha_opt.state_dict() if self.alpha_opt else None,
            "alpha_lr": self.alpha_opt.param_groups[0]['lr'] if self.alpha_opt else None,
            "cfg": self.cfg,
            "global_step": self._global_step,
        }
        th.save(checkpoint, path)

    def load(self, path: str, device: str):
        device_obj = th.device(device)
        checkpoint = th.load(path, map_location=device_obj, weights_only=False)
        self.cfg = checkpoint.get("cfg", self.cfg)
        self._global_step = checkpoint.get("global_step", 0)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt_state_dict"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt_state_dict"])

        loaded_log_alpha = checkpoint.get("log_alpha", None)
        if loaded_log_alpha is not None:
            self.log_alpha = loaded_log_alpha.to(device_obj)
            if self.alpha_opt is None:
                # Use saved learning rate if available, otherwise use cfg.lr
                saved_lr = checkpoint.get("alpha_lr", self.cfg.lr)
                self.alpha_opt = th.optim.Adam([self.log_alpha], lr=saved_lr)
            alpha_opt_state = checkpoint.get("alpha_opt_state_dict", None)
            if alpha_opt_state:
                self.alpha_opt.load_state_dict(alpha_opt_state)
            self.alpha_tensor = None
        else:
            self.log_alpha = None
            self.alpha_opt = None
            if self.alpha_tensor is None:
                self.alpha_tensor = th.tensor(0.2, device=device_obj)

        self.device = device_obj
        self.net.to(self.device)
        
        if self.alpha_tensor is not None:
            self.alpha_tensor = self.alpha_tensor.to(self.device)
