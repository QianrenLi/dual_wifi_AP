from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch as th
import torch.nn.functional as F
import importlib
import matplotlib.pyplot as plt

# Assuming these are defined elsewhere and necessary
from net_util.base import PolicyBase
from net_util.rnn_replay_with_pri_2 import RNNPriReplayBuffer2
from net_util.model.sac_rnn_belief_seq_dist import Network
from util.reward_to_value_bound import ValueDistribution
from . import register_policy, register_policy_cfg
from torch.utils.tensorboard import SummaryWriter

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
class SACRNNBeliefSeqDistV2_Config:
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
    annealing_max_lb: float = 1e-3
    annealing_epoch_max: float = 10


@register_policy
class SACRNNBeliefSeqDistV2(PolicyBase):
    """Refactored Soft Actor-Critic (SAC) with RNN and Learned Belief State."""

    def __init__(self, cmd_cls: Any, cfg: SACRNNBeliefSeqDistV2_Config, rollout_buffer: RNNPriReplayBuffer2 | None = None, device: str | None = None, state_transform_dict: dict | None = None, reward_cfg: Optional[dict] = None):
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
        self.critic_opt = th.optim.Adam(self.net.critic_parameters(), lr=cfg.lr * 5)
        self.belief_opt = th.optim.Adam(self.net.belief_parameters(), lr=cfg.lr * 5)
        
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

        self.vd = ValueDistribution(reward_cfg=self.reward_cfg, bins=bins)
        
        self._upd = 0               # number of critic updates
        self._global_step = 0       # training step index
        self._eval_h: Tensor | None = None
        self._belief_h: Tensor | None = None


    def _alpha(self) -> Tensor:
        """Current temperature α as a detached tensor."""
        return self.log_alpha.exp().detach() if self.log_alpha is not None else self.alpha_tensor.detach() # type: ignore


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
        """Writes all relevant metrics to the SummaryWriter (Helper)."""
        writer.add_scalar("loss/actor", kwargs['a_loss'].item(), step)
        writer.add_scalar("loss/critic", kwargs['c_loss'].item(), step)
        writer.add_scalar("loss/belief", kwargs['b_loss'].item(), step)
        writer.add_scalar("loss/KL", kwargs['belief_stats']['loss/KL'], step)
        
        if self.alpha_opt is not None:
            writer.add_scalar("loss/entropy", kwargs['ent_loss'].item(), step)
            writer.add_scalar("policy/alpha", self._alpha().item(), step)

        writer.add_scalar("q/qmin_pi", kwargs['qmin_pi'].mean().item(), step)
        writer.add_scalar("policy/logp_pi", kwargs['logp_TB1'].mean().item(), step)
    
    
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


    def _actor_loss(self, feat_TBH: Tensor, a_pi_TBA: Tensor, logp_TB1: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns: a_loss, qmin_pi (detached)"""
        alpha = self._alpha()
        q1_pi, q2_pi = self.net.critic_compute(feat_TBH, a_pi_TBA)
        
        q_val = th.min(self.vd.mean_value(q1_pi), self.vd.mean_value(q2_pi))
        
        a_loss = (alpha * logp_TB1 - q_val).mean()
        return a_loss, q_val


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

    def _probe_z_gradient_importance(
        self,
        obs_TBD: Tensor,
        act_TBA: Tensor,
        h_burn: Tensor,
        f_h_burn: Tensor,
        writer: Optional[SummaryWriter],
        step: int,
    ) -> None:
        """
        Gradient-based importance: how sensitive the mean Q is to obs vs z.

        We:
          - compute z from belief_encode (no grad)
          - treat obs and z as independent inputs to feature_compute with grad
          - compute mean Q under dataset actions
          - backprop to get ∂Q/∂obs and ∂Q/∂z
        Logs:
          z_importance/grad_norm_obs
          z_importance/grad_norm_z
          z_importance/grad_ratio_z   (z / (obs+z))
        """
        if writer is None:
            return

        self.net.zero_grad(set_to_none=True)

        # 1) Get z from belief encoder (no grad to avoid touching belief net)
        with th.no_grad():
            z_BH_raw, _, _, _ = self.net.belief_encode(obs_TBD, b_h=h_burn)

        # 2) Inputs for feature_compute with grad
        obs_feat = obs_TBD.detach().clone().requires_grad_(True)    # [T,B,D_obs]
        z_BH = z_BH_raw.detach().clone().requires_grad_(True)       # [B,H_latent] or compatible

        # 3) Features from obs & z
        feat_TBH, _ = self.net.feature_compute(obs_feat, z_BH, f_h_burn)

        # 4) Critic Q-values on dataset actions
        q_TB1K, q_TB1K2 = self.net.critic_compute(feat_TBH, act_TBA)         # [T,B,K]
        q_mean_TB1 = th.min(self.vd.mean_value(q_TB1K), self.vd.mean_value(q_TB1K2))          # [T,B,1] (your current API)
        scalar = -q_mean_TB1.mean()                                 # maximize Q -> minimize -Q

        scalar.backward()

        grad_obs_TBD = obs_feat.grad                                # [T,B,D_obs]
        grad_z_BH    = z_BH.grad                                    # [B,H_latent or ...]

        # If z is [B,H], broadcast it over T for a comparable norm
        if grad_z_BH is not None:
            grad_z_TBH = grad_z_BH
            grad_z_norm = grad_z_TBH.norm(dim=-1).mean().item()     # scalar
        else:
            grad_z_norm = 0.0

        grad_obs_norm = grad_obs_TBD.norm(dim=-1).mean().item()     # scalar

        ratio = grad_z_norm / (grad_z_norm + grad_obs_norm + 1e-8)

        writer.add_scalar("z_importance/grad_norm_obs", grad_obs_norm, step)
        writer.add_scalar("z_importance/grad_norm_z",   grad_z_norm,   step)
        writer.add_scalar("z_importance/grad_ratio_z",  ratio,         step)
        
        self.net.zero_grad(set_to_none=True)

    @th.no_grad()
    def _probe_z_weight_importance(
        self,
        writer: Optional[SummaryWriter],
        step: int,
    ) -> None:
        """
        Inspect first layer of feature extractor to see how much weight is on z vs obs.

        Assumes:
          self.net.fe.mlp[0] is nn.Linear with input dim = obs_dim + belief_dim

        Logs:
          z_importance/weight_norm_obs
          z_importance/weight_norm_z
          z_importance/weight_ratio_z   (z / (obs+z))
        """
        if writer is None:
            return

        fe = getattr(self.net, "fe", None)
        if fe is None or not hasattr(fe, "mlp"):
            return

        first = fe.mlp[0]
        if not isinstance(first, th.nn.Linear):
            return

        W = first.weight.data       # [H_out, D_in = obs_dim + belief_dim]

        obs_dim = int(self.cfg.obs_dim)
        z_dim   = int(self.cfg.belief_dim)
        if W.shape[1] < obs_dim + z_dim:
            # Safety check: unexpected shape, don't crash
            return

        W_obs = W[:, :obs_dim]
        W_z   = W[:, obs_dim:obs_dim + z_dim]

        norm_obs = W_obs.norm().item()
        norm_z   = W_z.norm().item()
        ratio    = norm_z / (norm_z + norm_obs + 1e-8)

        writer.add_scalar("z_importance/weight_norm_obs", norm_obs, step)
        writer.add_scalar("z_importance/weight_norm_z",   norm_z,   step)
        writer.add_scalar("z_importance/weight_ratio_z",  ratio,    step)

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
        y_hat_B1 = self.net.belief_decode( z_BH )
        
        b_loss, belief_stats = self._belief_loss( y_hat_B1, mu_BH, logvar_BH, interference.unsqueeze(-1), epoch )
        self._step_with_clip( self.net.belief_parameters(), self.belief_opt, b_loss, clip_norm=None)
        
        
        ## Alpha & Critic & Actor
        z_BH, _, _, _ = self.net.belief_encode( obs_train, b_h=h_burn )
        feat_TBH, _ = self.net.feature_compute( obs_train, z_BH, f_h_burn)
        a_pi_TBA, logp_TB1 = self.net.action_compute( feat_TBH )
        
        # Alpha update
        if self.alpha_opt is not None:
            ent_loss = self._alpha_loss(logp_TB1)
            self._step_with_clip( [self.log_alpha], self.alpha_opt, ent_loss, clip_norm=None ) # type: ignore
        
        # Critic update
        c_loss, c_loss_batch = self._critic_loss(
            feat_TBH.detach(), act_train, nxt_train, rew_train, done_train, importance_weights, b_h=h_burn, f_h=f_h_burn
        )
        self._step_with_clip(self.net.critic_parameters(), self.critic_opt, c_loss, clip_norm=None)
        self.net.soft_sync(self.cfg.tau)

        # Actor update
        with self.net.critics_frozen():
            a_loss, qmin_pi = self._actor_loss(feat_TBH, a_pi_TBA, logp_TB1)
            self._step_with_clip(self.net.actor_parameters(), self.actor_opt, a_loss, clip_norm=None)

        # Logging
        if self._global_step % self.cfg.log_interval == 0:
            self._log_batch_stats(writer, self._global_step, 
                a_loss=a_loss, c_loss=c_loss, b_loss=b_loss, ent_loss=ent_loss, logp_TB1=logp_TB1, 
                qmin_pi=qmin_pi, belief_stats=belief_stats, is_batch_rl=is_batch_rl
            )
            
        if self._global_step % (self.cfg.log_interval * 10) == 0:
            self._probe_z_ablation_feature(obs_train, h_burn, f_h_burn, writer, self._global_step)
            self._probe_z_gradient_importance(obs_train, act_train, h_burn, f_h_burn, writer, self._global_step)
            self._probe_z_weight_importance(writer, self._global_step)

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