from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch as th
import torch.nn.functional as F
import importlib

from net_util.base import PolicyBase
from net_util.replay.base import BaseReplayBuffer
from net_util.model.sac_rnn_belief_seq_dist import Network
from util.reward_to_value_bound import ValueDistribution
from .. import register_policy, register_policy_cfg
from torch.utils.tensorboard import SummaryWriter
# Assuming these are defined elsewhere and necessary

Tensor = th.Tensor

def symlog(x: th.Tensor, eps=1e-12) -> th.Tensor:
    return th.sign(x) * th.log(th.abs(x) + 1.0 + eps)

def _flatten_params(m: Optional[th.nn.Module]) -> th.Tensor:
    if m is None: return th.zeros(0)
    if isinstance(m , list):
        parts = [p.detach().view(-1).cpu() for p in m if p.requires_grad]
        return th.cat(parts) if parts else th.zeros(0)
    parts = [p.detach().view(-1).cpu() for p in m.parameters() if p.requires_grad]
    return th.cat(parts) if parts else th.zeros(0)

@register_policy_cfg
@dataclass
class SACRNNBeliefSeqDistV8_Config:
    # Core RL Parameters
    batch_size: int = 256
    learning_starts: int = 2_000
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    target_update_interval: int = 1
    
    # Entropy Regularization (SAC)
    ent_coef: str | float = "auto"     # "auto" or float (alpha)
    ent_coef_init: float = 1.0         # Initial alpha value if "auto"
    target_entropy: str | float = "auto" # "auto" -> -act_dim
    
    # Environment/Network Dimensions
    act_dim: int = 2
    obs_dim: int = 6
    belief_dim: int = 1
    
    # Device and Paths
    device: str = "cpu"
    seed: int = 0
    network_module_path: str = "net_util.model.sac_rnn_belief_seq_dist"
    load_path: str = "latest.pt"

    # Batch RL / Retraining
    batch_rl: bool = False
    retrain_interval: int = 0
    retrain_reset_alpha: bool = True
    retrain_reset_targets: bool = True

    # Critic UTD & Logging
    critic_utd: int = 1
    max_utd_ratio: float = 1.0
    log_interval: int = 50

    # CQL/CDL (for batch_rl)
    cdl_num_random: int = 10
    cdl_beta_cql_multiplier: float = 5 * 20 

    # Internal state helper for annealing beta
    annealing_max_lb: float = 2e-3
    annealing_epoch_max: float = 10
    
    # sigma_k
    sigma_k = 0.1


@register_policy
class SACRNNBeliefSeqDistV8(PolicyBase):
    """Refactored Soft Actor-Critic (SAC) with RNN and Learned Belief State."""

    def __init__(self, cmd_cls: Any, cfg: SACRNNBeliefSeqDistV8_Config, rollout_buffer: BaseReplayBuffer | None = None, device: str | None = None, state_transform_dict: dict | None = None, reward_cfg: Optional[dict] = None):
        super().__init__(cmd_cls, state_transform_dict=state_transform_dict, reward_cfg=reward_cfg)
        
        self.cfg = cfg
        self.device = th.device(device or cfg.device)
        th.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        th.backends.cudnn.benchmark = True
        
        bins = 51

        # Load Network (simplified)
        net_mod = importlib.import_module(cfg.network_module_path)
        net_class = getattr(net_mod, "Network")
        self.net: Network = net_class(cfg.obs_dim, cfg.act_dim, bins = bins, belief_dim=cfg.belief_dim).to(self.device)  # type: ignore

        self.buf = rollout_buffer

        # Optimizers (simplified actor LR setup)
        ac_lr = cfg.lr * (0.1 if cfg.batch_rl else 1.0)
        self.actor_opt = th.optim.Adam(self.net.actor_parameters(), lr=ac_lr)
        self.critic_opt = th.optim.Adam(self.net.critic_parameters(), lr=cfg.lr)
        self.belief_opt = th.optim.Adam(self.net.belief_parameters(), lr=cfg.lr)
        
        # Alpha Setup (consolidated)
        self.log_alpha, self.alpha_opt, self.alpha_tensor = None, None, None
        self.target_entropy: float | None = None
        if cfg.ent_coef == "auto" and not cfg.batch_rl:
            self.log_alpha = th.tensor([cfg.ent_coef_init], device=self.device).log().requires_grad_(True)
            self.alpha_opt = th.optim.Adam([self.log_alpha], lr=cfg.lr)
            self.target_entropy = -float(cfg.act_dim) if cfg.target_entropy == "auto" else float(cfg.target_entropy)
        else:
            alpha_val = float(cfg.ent_coef) if isinstance(cfg.ent_coef, float) else 0.2
            self.alpha_tensor = th.tensor(alpha_val, device=self.device)

        self.vd = ValueDistribution(reward_cfg=self.reward_cfg, bins=bins, gamma=self.cfg.gamma)
        
        self._upd = 0               # number of critic updates
        self._global_step = 0       # training step index
        self._eval_h: Tensor | None = None
        self._belief_h: Tensor | None = None


    def _alpha(self) -> Tensor:
        """Current temperature α as a detached tensor."""
        if self.log_alpha is not None:
            alpha = self.log_alpha.exp()
            return alpha.detach()
        else:
            return self.alpha_tensor.detach()


    def _get_beta(self, epoch: int) -> float:
        """
        Linear Warmup: Slowly introduces the information bottleneck.
        Helps prevent 'Posterior Collapse' where the model ignores the input
        and just predicts the mean interference (2.0).
        """
        # Number of epochs to reach full regularization (e.g., 50 or 100)
        warmup_steps = self.cfg.annealing_epoch_max 
        max_beta = self.cfg.annealing_max_lb
        
        if epoch >= warmup_steps:
            return max_beta
        
        # Linear ramp from 0.0 to max_beta
        return max(max_beta * (epoch / warmup_steps), 0)

    
    # @debug_param_change
    @staticmethod
    def _step_with_clip(params_iterable, opt: th.optim.Optimizer, loss: Tensor, clip_norm: float | None) -> None:
        """Shared helper for backward + (optional) grad clipping + step."""
        opt.zero_grad()
        loss.backward()
        if clip_norm is not None:
            th.nn.utils.clip_grad_norm_(params_iterable, clip_norm)
        opt.step()


    def _log_batch_stats(self, writer: SummaryWriter, step: int, **kwargs) -> None:
        writer.add_scalar("loss/actor", kwargs["a_loss"].item(), step)
        writer.add_scalar("loss/critic", kwargs["c_loss"].item(), step)
        writer.add_scalar("loss/belief", kwargs["b_loss"].item(), step)
        writer.add_scalar("loss/KL", kwargs["belief_stats"]["loss/KL"], step)

        if self.alpha_opt is not None:
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

    
    
    ## Loss
    def _belief_loss(
        self,
        y_hat_TBK: Tensor,   # [T, B, K] logits
        mu_TB1: Tensor,      # [T, B, 1]
        logvar_TB1: Tensor,  # [T, B, 1]
        interf_B1: Tensor,   # [B, 1] class index per batch element
        epoch: int
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Compute belief loss (CrossEntropy over K classes + beta * KL).
        """

        T, B, K = y_hat_TBK.shape

        # ------------------------------------------------------------------
        # 1) Repeat interf_B1 over time so it matches [T, B, 1]
        # ------------------------------------------------------------------
        # interf_B1: [B, 1] -> [1, B, 1] -> [T, B, 1]
        interf_TB1 = interf_B1.unsqueeze(0).expand(T, -1, -1)   # [T, B, 1]

        # ------------------------------------------------------------------
        # 2) Cross entropy loss over K classes
        #    Assume interf_* stores class indices in [0, K-1].
        # ------------------------------------------------------------------
        logits = y_hat_TBK.view(T * B, K)                       # [T*B, K]
        target = interf_TB1.squeeze(-1).long().view(T * B)      # [T*B]
        
        cross_entropy = F.cross_entropy(logits, target, reduction="mean")


        # ------------------------------------------------------------------
        # 3) KL term for the Gaussian latent
        # ------------------------------------------------------------------
        kl_loss = 0.5 * (mu_TB1.pow(2) + logvar_TB1.exp() - logvar_TB1 - 1.0).sum(-1).mean()
        
        

        beta = self._get_beta(epoch)
        b_loss = cross_entropy + beta * kl_loss

        stats = {
            "loss/CE": cross_entropy.item(),
            "loss/KL": kl_loss.item(),
            "belief/beta": beta,
        }

        return b_loss, stats


    def _critic_loss(
        self,
        feat_TBH: Tensor, act_TBA: Tensor, nxt_TBD: Tensor, rew_TB1: Tensor, done_TB1: Tensor, importance_weights: Tensor, b_h: Tensor | None = None, f_h: Tensor | None = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns: c_loss, c_loss_batch, cdl_stats"""

        backup_q_TB1 = self.net.target_backup_seq(
            nxt_TBD,
            rew_TB1,
            done_TB1,
            self.cfg.gamma,
            self._alpha(),
            self.vd,
            b_h,
            f_h
        )
        
        q1_pi, q2_pi = self.net.critic_compute(feat_TBH, act_TBA)     # [T,B,K]
        log_q1 = (q1_pi.clamp_min(1e-8)).log()
        log_q2 = (q2_pi.clamp_min(1e-8)).log()
        
        kl_loss = - (backup_q_TB1 * (log_q1)).sum(dim=-1) - (backup_q_TB1 * (log_q2)).sum(dim=-1)    # [T,B]
        c_loss_batch = kl_loss.mean(0)                      # [B]
        
        c_loss = (c_loss_batch * importance_weights).sum() / (importance_weights.sum() + 1e-8)

        return c_loss, c_loss_batch.detach()


    def _actor_loss(self, feat_TBH: Tensor, a_pi_TBA: Tensor, logp_TB1: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        alpha = self._alpha()
        q1_pi, q2_pi = self.net.critic_compute(feat_TBH, a_pi_TBA)
        # qmin_dist = th.minimum(q1_pi, q2_pi)
        qdist_avg = q1_pi.mean(dim=(0, 1)).detach()
        q_val = th.min(self.vd.mean_minus_k_sigma(q1_pi, self.cfg.sigma_k), self.vd.mean_value(q2_pi, self.cfg.sigma_k))
        a_loss = (alpha * logp_TB1 - q_val).mean()
        return a_loss, q_val.detach(), qdist_avg


    def _alpha_loss(self, logp_TB1: Tensor) -> Tensor:
        """Entropy temperature loss (0 if ent_coef is fixed)."""
        if self.alpha_opt is None or self.log_alpha is None:
            return th.zeros((), device=self.device)
        # logp is detached to prevent backprop into actor
        return -(self.log_alpha * (logp_TB1.detach() + self.target_entropy)).mean() # type: ignore


    ### Probe
    @th.no_grad()
    def _probe_z_ablation_feature(
        self,
        obs_TBD: Tensor,
        h_burn: Tensor,
        f_h_burn: Tensor,
        writer: Optional[SummaryWriter],
        step: int,
    ) -> None:
        """
        Measure importance of z by comparing features with real z vs zeroed z.

        Logs:
          z_importance/feat_delta_l2     : mean L2 change in feature
          z_importance/feat_rel_delta   : mean relative change in feature
        """
        if writer is None:
            return

        # 1) Get z from belief encoder (no grad, just a probe)
        z_BH, _, _, _ = self.net.belief_encode(obs_TBD, b_h=h_burn)

        # 2) Features with real z
        feat_full_TBH, _ = self.net.feature_compute(obs_TBD, z_BH, f_h_burn)  # [T,B,H]

        # 3) Features with zeroed z
        z_zero_BH = th.zeros_like(z_BH)
        feat_noz_TBH, _ = self.net.feature_compute(obs_TBD, z_zero_BH, f_h_burn)

        # 4) L2 distance in feature space as "impact" of z
        delta_TBH = feat_full_TBH - feat_noz_TBH                   # [T,B,H]
        imp_TB = delta_TBH.pow(2).sum(dim=-1).sqrt()               # [T,B]
        imp_scalar = imp_TB.mean().item()

        # Relative importance (normalized by feature norm)
        feat_norm_TB = feat_full_TBH.norm(dim=-1) + 1e-8           # [T,B]
        rel_imp_TB = (imp_TB / feat_norm_TB).mean().item()

        writer.add_scalar("z_importance/feat_delta_l2", imp_scalar, step)
        writer.add_scalar("z_importance/feat_rel_delta", rel_imp_TB, step)

        # Optional histogram for inspection
        writer.add_histogram("z_importance/feat_delta_l2_hist",
                             imp_TB.detach().cpu().view(-1), step)


    @th.no_grad()
    def _probe_z_ablation_action(
        self,
        obs_TBD: Tensor,
        h_burn: Tensor,
        f_h_burn: Tensor,
        writer: Optional[SummaryWriter],
        step: int,
        noise_std: float = 0.0,
    ) -> None:
        """
        Measure importance of z by comparing *actions* with real z vs ablated / perturbed z.

        Pipeline:
            z_BH           = belief_encode(obs_TBD, h_burn)
            feat_TBH       = feature_compute(obs_TBD, z_BH, f_h_burn)
            a_pi_TBA, _    = action_compute(feat_TBH, is_evaluate=True)

        Logs:
        z_importance/act_delta_l2_ablate     : mean L2 change in action (z vs zero-z)
        z_importance/act_rel_delta_ablate    : mean relative change in action
        z_importance/act_delta_l2_noise_*    : mean L2 change for z vs (z + noise)
        z_importance/act_rel_delta_noise_*   : mean relative change for noisy z
        """
        if writer is None:
            return

        # 1) Get z from belief encoder (no grad, just a probe)
        z_BH, _, _, _ = self.net.belief_encode(obs_TBD, b_h=h_burn)      # [T,B,H_z] or [B,H_z] depending on impl

        # 2) Features + actions with real z
        feat_full_TBH, _ = self.net.feature_compute(obs_TBD, z_BH, f_h_burn)  # [T,B,H]
        # Use deterministic / eval mode to reduce sampling noise
        a_full_TBA, _ = self.net.action_compute(feat_full_TBH, is_evaluate=True)  # [T,B,A]

        # ----------------- A) Pure ablation: z -> 0 ----------------- #
        z_zero_BH = th.zeros_like(z_BH)
        feat_noz_TBH, _ = self.net.feature_compute(obs_TBD, z_zero_BH, f_h_burn)
        a_noz_TBA, _ = self.net.action_compute(feat_noz_TBH, is_evaluate=True)

        # L2 distance in action space as "impact" of z
        delta_ablate_TBA = a_full_TBA - a_noz_TBA                     # [T,B,A]
        imp_ablate_TB = delta_ablate_TBA.pow(2).sum(dim=-1).sqrt()    # [T,B]
        imp_ablate_scalar = imp_ablate_TB.mean().item()

        # Relative importance (normalized by action norm)
        act_norm_TB = a_full_TBA.norm(dim=-1) + 1e-8                  # [T,B]
        rel_imp_ablate = (imp_ablate_TB / act_norm_TB).mean().item()

        writer.add_scalar("z_importance/act_delta_l2_ablate",
                        imp_ablate_scalar, step)
        writer.add_scalar("z_importance/act_rel_delta_ablate",
                        rel_imp_ablate, step)

        writer.add_histogram("z_importance/act_delta_l2_ablate_hist",
                            imp_ablate_TB.detach().cpu().view(-1), step)

        # ----------------- B) Perturbation: z -> z + noise ----------------- #
        if noise_std > 0.0:
            z_noise_BH = z_BH + noise_std * th.randn_like(z_BH)
            feat_noise_TBH, _ = self.net.feature_compute(obs_TBD, z_noise_BH, f_h_burn)
            a_noise_TBA, _ = self.net.action_compute(feat_noise_TBH, is_evaluate=True)

            delta_noise_TBA = a_full_TBA - a_noise_TBA                   # [T,B,A]
            imp_noise_TB = delta_noise_TBA.pow(2).sum(dim=-1).sqrt()     # [T,B]
            imp_noise_scalar = imp_noise_TB.mean().item()

            rel_imp_noise = (imp_noise_TB / act_norm_TB).mean().item()

            tag_suffix = f"{noise_std:g}"

            writer.add_scalar(f"z_importance/act_delta_l2_noise_{tag_suffix}",
                            imp_noise_scalar, step)
            writer.add_scalar(f"z_importance/act_rel_delta_noise_{tag_suffix}",
                            rel_imp_noise, step)

            writer.add_histogram(f"z_importance/act_delta_l2_noise_{tag_suffix}_hist",
                                imp_noise_TB.detach().cpu().view(-1), step)


    ## Train
    def train_per_epoch(
        self,
        epoch: int,
        writer: Optional[SummaryWriter] = None,
        is_batch_rl: bool | None = None,
        log_dir: str = "runs/sac_rnn",
    ) -> bool:
        """
        Main training loop for one epoch. All batch logic is inline here.
        """
        is_batch_rl = is_batch_rl if is_batch_rl is not None else self.cfg.batch_rl
        
        local_writer = writer is None
        writer = writer or SummaryWriter(log_dir=log_dir)
        
        ent_loss = None
        
        is_trained = False
        # --- Main Batch Loop ---
        burn_in = 50
        
        # 1. Get longer sequence
        obs_TBD, act_TBA, rew_TB1, nxt_TBD, done_TB1, info = next(self.buf.get_sequences(self.cfg.batch_size, trace_length=100))

        # Split data
        obs_burn = obs_TBD[:burn_in]
        obs_train = obs_TBD[burn_in:]
        act_train = act_TBA[burn_in:]
        rew_train = rew_TB1[burn_in:]
        nxt_train = nxt_TBD[burn_in:]
        done_train = done_TB1[burn_in:]

        # 2. Burn-in Phase (No Grads)
        with th.no_grad():
            # Start with zero state
            z_burn, h_burn, _, _ = self.net.belief_encode(obs_burn) 
            _, f_h_burn = self.net.feature_compute( obs_burn, z_burn.detach() )

        is_trained = True
        importance_weights:Tensor = info["is_weights"].squeeze(-1) # [B]
        interference:Tensor = info["interference"].squeeze(-1) # [B]
        
        ## Belief
        z_BH, _, mu_BH, logvar_BH = self.net.belief_encode(obs_train, b_h=h_burn)
        
        ## Alpha & Critic & Actor
        feat_TBH, _ = self.net.feature_compute( obs_train, z_BH.detach(), f_h_burn)
        a_pi_TBA, logp_TB1 = self.net.action_compute( feat_TBH )
        
        # Alpha update
        if self.alpha_opt is not None and self._global_step % 5 == 0:
            ent_loss = self._alpha_loss(logp_TB1)
            self._step_with_clip( [self.log_alpha], self.alpha_opt, ent_loss, clip_norm=5.0 ) # type: ignore
        
        # Critic update
        c_loss, c_loss_batch = self._critic_loss(
            feat_TBH, act_train, nxt_train, rew_train, done_train, importance_weights, b_h=h_burn, f_h=f_h_burn
        )
        self._step_with_clip(self.net.critic_parameters(), self.critic_opt, c_loss, clip_norm=10.0)
        self.net.soft_sync(self.cfg.tau)

        # Actor update
        feat_TBH, _ = self.net.feature_compute(obs_train, z_BH.detach(), f_h_burn)
        a_pi_TBA, logp_TB1, mu_TBA, std_TBA = self.net.action_compute(feat_TBH, return_stats=True)
        with self.net.critics_frozen():
            a_loss, qmin_pi, qdist_avg = self._actor_loss(feat_TBH, a_pi_TBA, logp_TB1)
            self._step_with_clip(self.net.actor_parameters(), self.actor_opt, a_loss, clip_norm=5.0)

            mu_mean = mu_TBA.mean(dim=(0, 1))
            mu_std = mu_TBA.std(dim=(0, 1))
            sigma_mean = std_TBA.mean(dim=(0, 1))
            sigma_std = std_TBA.std(dim=(0, 1))
            act_mean = a_pi_TBA.mean(dim=(0, 1))
            act_std = a_pi_TBA.std(dim=(0, 1))

        if self._global_step % 5 == 0:
            y_hat_B1 = self.net.belief_decode( z_BH )
            b_loss, belief_stats = self._belief_loss( y_hat_B1, mu_BH, logvar_BH, interference.unsqueeze(-1), epoch )
            self._step_with_clip( self.net.belief_parameters(), self.belief_opt, b_loss, clip_norm=5.0)

            if self._global_step % self.cfg.log_interval == 0:
                self._log_batch_stats(
                    writer,
                    self._global_step,
                    a_loss=a_loss,
                    c_loss=c_loss,
                    b_loss=b_loss,
                    ent_loss=ent_loss,
                    logp_TB1=logp_TB1,
                    qmin_pi=qmin_pi,
                    belief_stats=belief_stats,
                    is_batch_rl=is_batch_rl,
                    qdist_avg=qdist_avg,
                    mu_mean=mu_mean,
                    mu_std=mu_std,
                    sigma_mean=sigma_mean,
                    sigma_std=sigma_std,
                    act_mean=act_mean,
                    act_std=act_std,
                )

            
        if self._global_step % (self.cfg.log_interval * 10) == 0:
            self._probe_z_ablation_feature(obs_train, h_burn, f_h_burn, writer, self._global_step)
            self._probe_z_ablation_action(obs_train, h_burn, f_h_burn, writer, self._global_step, noise_std=0.1)


        self._global_step += 1
        # self.buf.update_episode_losses(info["ep_ids"], c_loss_batch.cpu().numpy()) # type: ignore

        if local_writer: writer.flush(); writer.close()
        return is_trained

    # --- Inference / Save / Load (Unchanged) ---
    @th.no_grad()
    def tf_act(self, obs_vec: list[float], is_evaluate: bool = False, reset_hidden: bool = False) -> Dict[str, np.ndarray]:
        # Logic remains the same: load obs, init/update hidden/belief, sample/evaluate, return results.
        obs = th.tensor(obs_vec, device=self.device, dtype=th.float32).unsqueeze(0).unsqueeze(0)

        z_BH, _belief_h, _, _ = self.net.belief_encode(obs, self._belief_h, is_evaluate=is_evaluate)
        y_hat_B1 = self.net.belief_decode(z_BH)
        belief = (F.softmax(y_hat_B1, dim=-1) * th.arange(y_hat_B1.shape[-1], device=self.device)).sum(dim=-1, keepdim=True)

        feat, _eval_h = self.net.feature_compute(obs, z_BH, self._eval_h)
        
        action, logp = self.net.action_compute(feat, is_evaluate)
        v_like = 0

        self._belief_h = _belief_h.detach()
        self._eval_h = _eval_h.detach()

        return {
            "action": action[0][0].cpu().numpy(),
            "log_prob": logp[0][0].cpu().numpy(),
            "value": v_like,
            "belief": belief[0][0].cpu().numpy(),
        }


    def save(self, path: str):
        # ... (save logic is the same)
        checkpoint = {
            "model_state_dict": self.net.state_dict(),
            "actor_opt_state_dict": self.actor_opt.state_dict(),
            "critic_opt_state_dict": self.critic_opt.state_dict(),
            "belief_opt_state_dict": self.belief_opt.state_dict(),
            "log_alpha": self.log_alpha,
            "alpha_opt_state_dict": self.alpha_opt.state_dict() if self.alpha_opt else None,
            "cfg": self.cfg,
            "global_step": self._global_step,
        }
        th.save(checkpoint, path)

    def load(self, path: str, device: str):
        # ... (load logic is the same)
        device_obj = th.device(device)
        checkpoint = th.load(path, map_location=device_obj, weights_only=False)
        self.cfg = checkpoint.get("cfg", self.cfg)
        self._global_step = checkpoint.get("global_step", 0)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.actor_opt.load_state_dict(checkpoint["actor_opt_state_dict"])
        self.critic_opt.load_state_dict(checkpoint["critic_opt_state_dict"])
        self.belief_opt.load_state_dict(checkpoint["belief_opt_state_dict"])

        loaded_log_alpha = checkpoint.get("log_alpha", None)
        if loaded_log_alpha is not None:
            self.log_alpha = loaded_log_alpha.to(device_obj)
            if self.alpha_opt is None:
                self.alpha_opt = th.optim.Adam([self.log_alpha], lr=self.cfg.lr)
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