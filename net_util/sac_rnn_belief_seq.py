from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch as th
import torch.nn.functional as F
import importlib

# Assuming these are defined elsewhere and necessary
from net_util.base import PolicyBase
from net_util.rnn_replay_with_pri_2 import RNNPriReplayBuffer2
from . import register_policy, register_policy_cfg
from torch.utils.tensorboard import SummaryWriter

Tensor = th.Tensor

@register_policy_cfg
@dataclass
class SACRNNBeliefSeq_Config:
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
    network_module_path: str = "net_util.model.sac_rnn_belief"
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
    annealing_max_lb: float = 2.0
    annealing_epoch_max: float = 5e3


@register_policy
class SACRNNBeliefSeq(PolicyBase):
    """Refactored Soft Actor-Critic (SAC) with RNN and Learned Belief State."""

    def __init__(self, cmd_cls: Any, cfg: SACRNNBeliefSeq_Config, rollout_buffer: RNNPriReplayBuffer2 | None = None, device: str | None = None, state_transform_dict: dict | None = None):
        super().__init__(cmd_cls, state_transform_dict=state_transform_dict)
        
        self.cfg = cfg
        self.device = th.device(device or cfg.device)
        th.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        th.backends.cudnn.benchmark = True

        # Load Network (simplified)
        net_mod = importlib.import_module(cfg.network_module_path)
        Network = getattr(net_mod, "Network")
        self.net = Network(cfg.obs_dim, cfg.act_dim, belief_dim=cfg.belief_dim).to(self.device)  # type: ignore

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

        self._upd = 0               # number of critic updates
        self._global_step = 0       # training step index
        self._eval_h: Tensor | None = None
        self._belief_h: Tensor | None = None


    def _alpha(self) -> Tensor:
        """Current temperature α as a detached tensor."""
        return self.log_alpha.exp().detach() if self.log_alpha is not None else self.alpha_tensor.detach() # type: ignore


    def _get_beta(self, epoch: int) -> float:
        """Belief loss KL annealing parameter."""
        max_lb = self.cfg.annealing_max_lb
        return 0.01 + 0.5 * (max_lb - 0.01) * (1 + np.sin(np.pi * epoch / self.cfg.annealing_epoch_max))

 
    @staticmethod
    def _step_with_clip(params_iterable, opt: th.optim.Optimizer, loss: Tensor, clip_norm: float | None) -> None:
        """Shared helper for backward + (optional) grad clipping + step."""
        opt.zero_grad(set_to_none=True)
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

        writer.add_scalar("q/q1_mean", kwargs['q1_TB1'].mean().item(), step)
        writer.add_scalar("q/qmin_pi", kwargs['qmin_pi'].mean().item(), step)
        writer.add_scalar("policy/logp_pi", kwargs['logp_TB1'].mean().item(), step)
        
        if kwargs['is_batch_rl'] and kwargs['cdl_stats'] is not None:
            for k, v in kwargs['cdl_stats'].items():
                writer.add_scalar(k, v.item(), step)


    ## Loss
    def _belief_loss(self, obs_TBD: Tensor, info: Dict[str, Tensor], epoch: int) -> Tuple[Tensor, Dict[str, Any], Tensor]:
        """
        Compute belief loss (MSE + beta * KL).
        Returns: b_loss, stats, belief_seq_TB1_sac (detached z)
        """
        B = obs_TBD.shape[1]
        device = obs_TBD.device

        # Fresh zero hidden state for belief RNN training
        belief_h0 = self.net.belief_rnn.init_state(B, device=device)
        z_TB1, _, mu_TB1, logvar_TB1, y_hat_TB1 = self.net.belief_predict_seq(obs_TBD, belief_h0)

        # Last step supervision
        mu_last_BH = mu_TB1[-1]
        logv_last_BH = logvar_TB1[-1]
        y_last_B1 = y_hat_TB1[-1]
        
        interf_B1 = info["interference"].to(device)
        iw_B1 = info["is_weights"].detach().to(device)

        # 1. MSE Loss (L1)
        mse_loss = (y_last_B1 - interf_B1).pow(2).mean()

        # 2. KL Loss (D_KL[N(mu, logvar) || N(0, I)])
        kl_loss = 0.5 * (mu_last_BH.pow(2) + logv_last_BH.exp() - logv_last_BH - 1.0).mean()

        # Total belief loss with annealing
        beta = self._get_beta(epoch)
        b_loss = mse_loss + beta * kl_loss

        stats = {
            "loss/KL": kl_loss.item(),
            "loss/MSE": mse_loss.item(),
            "belief/beta": beta,
        }

        # Use z for RL path (detached)
        return b_loss, z_TB1.detach(), stats

    def _cdl_loss(
        self, feat_TBH: Tensor, nxt_TBD: Tensor, belief_seq_TB1_sac: Tensor, act_TBA: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Conservative Dual Loss (CDL) for Batch RL."""
        T, B, H = feat_TBH.shape
        _, _, A = act_TBA.shape
        device = feat_TBH.device
        N_rand = self.cfg.cdl_num_random
        beta_cql = self.cfg.cdl_beta_cql_multiplier / self.buf.sigma # type: ignore

        # Helper to evaluate Q for [T, B, N, A] actions
        def _eval_q_proposals(feat_TBH_: Tensor, act_TBNA_: Tensor):
            T_, B_, N_, A_ = act_TBNA_.shape
            feat_rep = feat_TBH_.unsqueeze(2).expand(T_, B_, N_, H).reshape(-1, H)
            act_rep = act_TBNA_.reshape(-1, A_)
            q1f, q2f = self.net.q(feat_rep, act_rep)
            return q1f.view(T_, B_, N_, 1), q2f.view(T_, B_, N_, 1)

        with th.no_grad():
            # 1. Random actions: [T, B, N_rand, A]
            a_rand_TBNA = th.rand(T, B, N_rand, A, device=device) * 2 - 1

            # Get policy actions: π(s_t) and π(s_{t+1})
            a_curr_TBA, _ = self.net.sample_from_features(feat_TBH, detach_feat_for_actor=True)
            
            h0_enc, _ = self.net.init_hidden(B, device)
            feat_next_TBH, _ = self.net.encode_seq(nxt_TBD, belief_seq_TB1_sac, h0_enc)
            a_next_TBA, _ = self.net.sample_from_features(feat_next_TBH, detach_feat_for_actor=True)

            # Expand policy actions to [T, B, N_rand, A]
            a_curr_TBNA = a_curr_TBA.unsqueeze(2).expand(T, B, N_rand, A)
            a_next_TBNA = a_next_TBA.unsqueeze(2).expand(T, B, N_rand, A)

        # Q-values for proposal actions
        q1_rand, q2_rand = _eval_q_proposals(feat_TBH, a_rand_TBNA)
        q1_curr, q2_curr = _eval_q_proposals(feat_TBH, a_curr_TBNA)
        q1_next, q2_next = _eval_q_proposals(feat_TBH, a_next_TBNA)

        # Q-values for behavioral (data) action
        q1_bc, q2_bc = self.net.q(feat_TBH, act_TBA)  # [T,B,1]

        # Concatenate Q-values (N_total = 3*N_rand + 1)
        q1_proposals = th.cat([q1_rand, q1_bc.unsqueeze(2), q1_next, q1_curr], dim=2)
        q2_proposals = th.cat([q2_rand, q2_bc.unsqueeze(2), q2_next, q2_curr], dim=2)

        # LSE penalties: (LSE - Q_behavioral)
        lse1_TB = th.logsumexp(q1_proposals.squeeze(-1), dim=2)
        lse2_TB = th.logsumexp(q2_proposals.squeeze(-1), dim=2)
        
        cql1 = lse1_TB.mean() - q1_bc.mean()
        cql2 = lse2_TB.mean() - q2_bc.mean()
        penalty = beta_cql * (cql1 + cql2)

        # Stats for logging
        min_q_rand = th.minimum(q1_rand, q2_rand).mean()
        min_q_curr = th.minimum(q1_curr, q2_curr).mean()
        min_q_bc = th.minimum(q1_bc, q2_bc).mean()

        stats = {
            "cdl/penalty_total": penalty.detach(),
            "cdl/gap_rand_bc": (min_q_rand - min_q_bc).detach(),
            "cdl/gap_curr_bc": (min_q_curr - min_q_bc).detach(),
        }
        return penalty, stats


    def _critic_loss(
        self,
        feat_TBH: Tensor, act_TBA: Tensor, nxt_TBD: Tensor,
        belief_seq_TB1_sac: Tensor, hT: Tensor,
        rew_TB1: Tensor, done_TB1: Tensor, importance_weights: Tensor,
        is_batch_rl: bool,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor] | None]:
        """Returns: c_loss, c_loss_batch, q1_TB1, q2_TB1, cdl_stats"""
        alpha = self._alpha()
        T, B, _ = belief_seq_TB1_sac.shape
        device = nxt_TBD.device

        with th.no_grad():
            belief_h0_next = self.net.belief_rnn.init_state(B, device=device)
            belief_next_TB1_sac, _, _, _, _ = self.net.belief_predict_seq(nxt_TBD, belief_h0_next)

            backup_TB1 = self.net.target_backup_seq(
                nxt_TBD,
                belief_next_TB1_sac.detach(),
                hT,
                rew_TB1,
                done_TB1,
                self.cfg.gamma,
                alpha,
            )

        q1_TB1, q2_TB1 = self.net.q(feat_TBH, act_TBA)  # [T,B,1]

        diff = th.stack([q1_TB1, q2_TB1], dim=0) - backup_TB1
        # diff = diff / self.buf.sigma  # type: ignore
        c_loss_per_t = diff.pow(2).mean(dim=0)

        c_loss_batch = c_loss_per_t.mean(dim=(0, 2))  # [B]
        c_loss = (c_loss_batch * importance_weights).mean()

        cdl_stats = None
        if is_batch_rl:
            cdl_penalty, cdl_stats = self._cdl_loss(feat_TBH, nxt_TBD, belief_next_TB1_sac.detach(), act_TBA)
            c_loss = c_loss + cdl_penalty

        return c_loss, c_loss_batch.detach(), q1_TB1, q2_TB1, cdl_stats


    def _actor_loss(self, feat_TBH: Tensor, a_pi_TBA: Tensor, logp_TB1: Tensor) -> Tuple[Tensor, Tensor]:
        """Returns: a_loss, qmin_pi (detached)"""
        alpha = self._alpha()
        q1_pi, q2_pi = self.net.q(feat_TBH.detach(), a_pi_TBA)
        qmin_pi = th.min(q1_pi, q2_pi)
        a_loss = (alpha * logp_TB1 - qmin_pi).mean()
        return a_loss, qmin_pi.detach()

    def _alpha_loss(self, logp_TB1: Tensor) -> Tensor:
        """Entropy temperature loss (0 if ent_coef is fixed)."""
        if self.alpha_opt is None or self.log_alpha is None:
            return th.zeros((), device=self.device)
        # logp is detached to prevent backprop into actor
        return -(self.log_alpha * (logp_TB1.detach() + self.target_entropy)).mean() # type: ignore
    
    def train_per_epoch(
        self,
        epoch: int,
        writer: Optional[SummaryWriter] = None,
        log_dir: str = "runs/sac_rnn",
        is_batch_rl: bool | None = None,
    ) -> bool:
        """
        Main training loop for one epoch. All batch logic is inline here.
        """
        is_batch_rl = is_batch_rl if is_batch_rl is not None else self.cfg.batch_rl
        
        local_writer = writer is None
        writer = writer or SummaryWriter(log_dir=log_dir)
        
        data_num = getattr(self.buf, "data_num", 0)
        ent_loss = None
        
        # Early stop checks
        utd_ratio = self._upd / max(float(data_num), 1.0)
        if data_num <= self.cfg.batch_size or (not is_batch_rl and utd_ratio >= self.cfg.max_utd_ratio):
            if local_writer: writer.flush(); writer.close()
            return False

        is_trained = False
        # --- Main Batch Loop ---
        for batch in self.buf.get_sequences(self.cfg.batch_size, trace_length=100, device=self.device):
            is_trained = True
            obs_TBD, act_TBA, rew_TB1, nxt_TBD, done_TB1, info = batch
            B = obs_TBD.shape[1]
            device = obs_TBD.device

            importance_weights = info["is_weights"].detach().to(device).squeeze(-1)
            h0, _ = self.net.init_hidden(B, device)

            # 1. Belief Update
            b_loss, belief_seq_TB1_sac, belief_stats = self._belief_loss(obs_TBD, info, epoch)
            self._step_with_clip(self.net.belief_parameters(), self.belief_opt, b_loss, clip_norm=5.0)

            # 3. Critic Updates (with UTD)
            c_loss, c_loss_batch, q1_TB1, q2_TB1, cdl_stats = None, None, None, None, None
            for _ in range(max(1, self.cfg.critic_utd)):
                feat_TBH, hT = self.net.encode_seq(obs_TBD, belief_seq_TB1_sac, h0)
                c_loss, c_loss_batch, q1_TB1, q2_TB1, cdl_stats = self._critic_loss(
                    feat_TBH, act_TBA, nxt_TBD, belief_seq_TB1_sac, hT,
                    rew_TB1, done_TB1, importance_weights, is_batch_rl,
                )
                self._step_with_clip(self.net.critic_parameters(), self.critic_opt, c_loss, clip_norm=10.0)
                self.net._soft_sync(self.cfg.tau)
                self._upd += 1

            # Update priorities using per-episode critic loss
            self.buf.update_episode_losses(info["ep_ids"], c_loss_batch.cpu().numpy()) # type: ignore

            # 4. Actor and Alpha Updates
            feat_TBH, hT = self.net.encode_seq(obs_TBD, belief_seq_TB1_sac, h0)
            a_pi_TBA, logp_TB1 = self.net.sample_from_features(feat_TBH)
            a_loss, qmin_pi = self._actor_loss(feat_TBH, a_pi_TBA, logp_TB1)
            self._step_with_clip(self.net.actor_parameters(), self.actor_opt, a_loss, clip_norm=5.0)

            if self.alpha_opt is not None:
                ent_loss = self._alpha_loss(logp_TB1)
                self._step_with_clip([self.log_alpha], self.alpha_opt, ent_loss, clip_norm=None) # type: ignore

            # 5. Logging
            if self._global_step % self.cfg.log_interval == 0:
                self._log_batch_stats(writer, self._global_step, 
                    a_loss=a_loss, c_loss=c_loss, b_loss=b_loss, ent_loss=ent_loss, logp_TB1=logp_TB1, 
                    q1_TB1=q1_TB1, qmin_pi=qmin_pi, cdl_stats=cdl_stats, belief_stats=belief_stats, is_batch_rl=is_batch_rl
                )

            self._global_step += 1

        # Reset hidden states after epoch
        self._belief_h = None   
        
        if local_writer: writer.flush(); writer.close()
        return is_trained

    # --- Inference / Save / Load (Unchanged) ---
    @th.no_grad()
    def tf_act(self, obs_vec: list[float], is_evaluate: bool = False, reset_hidden: bool = False) -> Dict[str, np.ndarray]:
        # Logic remains the same: load obs, init/update hidden/belief, sample/evaluate, return results.
        obs = th.tensor(obs_vec, device=self.device, dtype=th.float32).unsqueeze(0)
        if reset_hidden or self._eval_h is None or self._belief_h is None:
            self._eval_h, self._belief_h = self.net.init_hidden(1, self.device)
        z_B1, h1, _, _, y_hat_B1 = self.net.belief_predict_step(obs, self._belief_h)

        feat, h_next = self.net.encode(obs, z_B1.detach(), self._eval_h)
        
        if is_evaluate:
            mu, _ = self.net._mean_std(feat)
            action = th.tanh(mu)
            q1, q2 = self.net.q(feat, action)
            logp = th.zeros(1, 1, device=self.device)
            v_like = th.min(q1, q2)[0].detach().cpu().numpy()
        else:
            action, logp = self.net.sample_from_features(feat, detach_feat_for_actor=True)
            v_like = 0
        
        self._belief_h = h1.detach()
        self._eval_h = h_next.detach()

        return {
            "action": action[0].detach().cpu().numpy(),
            "log_prob": logp[0].detach().cpu().numpy(),
            "value": v_like,
            "belief": y_hat_B1[0].detach().cpu().numpy(),
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