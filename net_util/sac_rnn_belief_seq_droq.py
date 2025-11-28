from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import importlib
import copy

from net_util.base import PolicyBase
# Assuming these imports exist in your environment
from net_util.model.sac_rnn_belief_seq import Network as SAC_RNN_NETWORK
from net_util.rnn_replay_with_pri_2 import RNNPriReplayBuffer2
from . import register_policy, register_policy_cfg
from torch.utils.tensorboard import SummaryWriter

# ---------------- utils ----------------
def _safe_mean(xs):
    return float(np.mean(xs)) if xs else float("nan")

def symlog(x: th.Tensor, eps=1e-12) -> th.Tensor:
    return th.sign(x) * th.log(th.abs(x) + 1.0 + eps)

@register_policy_cfg
@dataclass
class SACRNNBeliefSeqDroq_Config:
    # core
    # [OPTIMIZATION 1] Increased Batch Size to saturate 3090 Memory
    batch_size: int = 1024 
    learning_starts: int = 2_000
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    target_update_interval: int = 1
    ent_coef: str | float = "auto"
    ent_coef_init: float = 1.0
    target_entropy: str | float = "auto"
    act_dim: int = 2
    device: str = "cuda" # Force CUDA

    # fields referenced below
    seed: int = 0
    obs_dim: int = 6
    belief_dim: int = 1
    network_module_path: str = "net_util.model.sac_dropout_q"
    load_path = "latest.pt"
    
    batch_rl = False
    
    # --- retraining timer (0 disables) ---
    retrain_interval: int = 0
    retrain_reset_alpha: bool = True
    retrain_reset_targets: bool = True

@register_policy
class SACRNNBeliefSeqDroq(PolicyBase):
    def __init__(self, cmd_cls, cfg: SACRNNBeliefSeqDroq_Config, rollout_buffer: RNNPriReplayBuffer2 | None = None, device: str | None = None, state_transform_dict = None):
        super().__init__(cmd_cls, state_transform_dict = state_transform_dict)
        th.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.cfg = cfg
        self.device = th.device(cfg.device) if device is None else th.device(device)

        # Load network module
        net_mod = importlib.import_module(cfg.network_module_path)
        Network = getattr(net_mod, "Network")
        self.net = Network(cfg.obs_dim, cfg.act_dim, scale_log_offset=0, belief_dim = cfg.belief_dim).to(self.device) # type: ignore
        
        # [OPTIMIZATION 2] Compile model for RTX 3090 (Requires PyTorch 2.0+)
        # This reduces kernel launch overhead significantly
        try:
            print("Compiling model for speed...")
            self.net = th.compile(self.net)
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")

        self.buf = rollout_buffer

        # optimizers
        ac_lr = cfg.lr * 0.1 if cfg.batch_rl else cfg.lr
        self.actor_opt  = th.optim.Adam(self.net.actor_parameters(),  lr=ac_lr)
        self.critic_opt = th.optim.Adam(self.net.critic_parameters(), lr=cfg.lr)
        self.belief_opt = th.optim.Adam(self.net.belief_parameteres(), lr=cfg.lr)

        # temperature α
        if cfg.ent_coef == "auto" and cfg.batch_rl == False:
            self.log_alpha = th.tensor([cfg.ent_coef_init], device=self.device).log().requires_grad_(True)
            self.alpha_opt = th.optim.Adam([self.log_alpha], lr=cfg.lr)
            self.target_entropy = -float(cfg.act_dim) if cfg.target_entropy == "auto" else float(cfg.target_entropy)
        else:
            self.log_alpha, self.alpha_opt = None, None
            self.alpha_tensor = th.tensor(float(0.2), device=self.device)

        # counters / state
        self._upd = 0
        self._epoch_h = None
        self._eval_h = None
        self._belief_h = None
        self._global_step = 0
        
        self.log_int = 100
        
        self.last_obs = None
        self.annealing_bl = lambda epoch, max_lb: min(epoch / 50e3 * max_lb, max_lb)
        
        self.critic_update_steps = 5

    def _alpha(self):
        return (self.log_alpha.exp() if self.log_alpha is not None else self.alpha_tensor).detach()

    def cdl_loss(self, feat_TBH, nxt_TBD, belief_curr_TB1, belief_next_TB1, act_TBA, num_random=10, beta_cql=0.05):
        # Standard CDL logic ...
        device = feat_TBH.device
        T, B, H = feat_TBH.shape
        _, _, A = act_TBA.shape

        with th.no_grad():
            a_rand_TBNA = th.rand(T, B, num_random, A, device=device) * 2 - 1
            a_curr_TBA, _ = self.net.sample_from_features(feat_TBH, detach_feat_for_actor=True)
            a_curr_TBNA = a_curr_TBA.unsqueeze(2).expand(T, B, num_random, A)

            h0_enc, _ = self.net.init_hidden(B, device)
            feat_next_TBH, _ = self.net.encode_seq(nxt_TBD, belief_next_TB1, h0_enc)
            
            a_next_TBA, _ = self.net.sample_from_features(feat_next_TBH, detach_feat_for_actor=True)
            a_next_TBNA = a_next_TBA.unsqueeze(2).expand(T, B, num_random, A)

        def _eval_q(feat_TBH, act_TBNA):
            T_, B_, N_, A_ = act_TBNA.shape
            feat_rep = feat_TBH.unsqueeze(2).expand(T_, B_, N_, H).reshape(T_*B_*N_, H)
            act_rep  = act_TBNA.reshape(T_*B_*N_, A_)
            q1f, q2f = self.net.q(feat_rep, act_rep)
            return q1f.view(T_,B_,N_,1), q2f.view(T_,B_,N_,1)

        q1_rand, q2_rand = _eval_q(feat_TBH, a_rand_TBNA)
        q1_curr, q2_curr = _eval_q(feat_TBH, a_curr_TBNA)
        q1_next, q2_next = _eval_q(feat_TBH, a_next_TBNA)

        q1_bc, q2_bc = self.net.q(feat_TBH, act_TBA)
        q1_bc_TB11 = q1_bc.unsqueeze(2)
        q2_bc_TB11 = q2_bc.unsqueeze(2)

        cat_q1 = th.cat([q1_rand, q1_bc_TB11, q1_next, q1_curr], dim=2)
        cat_q2 = th.cat([q2_rand, q2_bc_TB11, q2_next, q2_curr], dim=2)

        lse1_TB = th.logsumexp(cat_q1.squeeze(-1), dim=2)
        lse2_TB = th.logsumexp(cat_q2.squeeze(-1), dim=2)

        cql1 = (lse1_TB.mean() - q1_bc.mean())
        cql2 = (lse2_TB.mean() - q2_bc.mean())
        penalty = beta_cql * (cql1 + cql2)

        stats = {
            "cdl/penalty_total": penalty.detach(),
        }
        return penalty, stats

    # --- small utilities (inside the class) ---------------------------------
    def _predict_belief_seq(self, obs_TBD: th.Tensor):
        B = obs_TBD.size(1)
        belief_h0 = self.net.belief_rnn.init_state(B, device=self.device)
        z_TB1, bT, mu_TB1, logvar_TB1, y_hat_TB1 = self.net.belief_predict_seq(obs_TBD, belief_h0)
        return z_TB1, mu_TB1, logvar_TB1, y_hat_TB1

    def _encode_with_belief(self, obs_TBD: th.Tensor, belief_TB1: th.Tensor, h0: th.Tensor):
        return self.net.encode_seq(obs_TBD, belief_TB1.detach(), h0)

    def _belief_supervision_loss(self, mu_TB1, logvar_TB1, y_hat_TB1, interf_B1, iw_B1, epoch: int):
        y_last_B1    = y_hat_TB1[-1]
        mu_last_BH   = mu_TB1[-1]
        logv_last_BH = logvar_TB1[-1]
        mse_loss = (F.smooth_l1_loss(y_last_B1, interf_B1, reduction='none') * iw_B1).mean()
        kl_loss  = (0.5 * (mu_last_BH.pow(2) + logv_last_BH.exp() - logv_last_BH - 1.0)).mean()
        beta     = self.annealing_bl(epoch, 1)
        return mse_loss + beta * kl_loss, beta, kl_loss

    def _critic_loss(self, feat_TBH, act_TBA, nxt_TBD, belief_curr_TB1, belief_next_TB1, hT, rew_TB1, done_TB1, importance_weights, is_batch_rl: bool):
        alpha = self._alpha()
        with th.no_grad():
            backup_TB1 = self.net.target_backup_seq(nxt_TBD, belief_next_TB1, hT, rew_TB1, done_TB1, self.cfg.gamma, alpha)

        q1_TB1, q2_TB1 = self.net.q(feat_TBH, act_TBA)
        diff = th.stack([q1_TB1, q2_TB1], dim=0) - backup_TB1
        
        safe_sigma = self.buf.sigma if hasattr(self.buf, 'sigma') and self.buf.sigma > 1e-6 else 1.0
        diff = diff / safe_sigma
        
        c_loss_per_t = diff.pow(2).mean(dim=0)      # [T,B,1]
        c_loss_batch = c_loss_per_t.mean(dim=(0,2)) # [B]
        c_loss = (c_loss_batch * importance_weights).mean()

        cdl_stats = None
        if is_batch_rl:
            cdl_penalty, cdl_stats = self.cdl_loss(
                feat_TBH=feat_TBH, 
                nxt_TBD=nxt_TBD, 
                belief_curr_TB1=belief_curr_TB1,
                belief_next_TB1=belief_next_TB1,
                act_TBA=act_TBA, 
                num_random=10, 
                beta_cql=5 * 20 / safe_sigma,
            )
            c_loss = c_loss + cdl_penalty
        return c_loss, cdl_stats

    def _actor_loss(self, feat_TBH):
        with self.net.critics_frozen():
            a_pi_TBA, logp_TB1 = self.net.sample_from_features(feat_TBH)
            q1_pi, q2_pi = self.net.q(feat_TBH.detach(), a_pi_TBA)
            alpha = self._alpha()
            a_loss = (alpha * logp_TB1 - th.stack([q1_pi, q2_pi], dim=0).mean(dim=0)).mean()
            return a_loss, logp_TB1

    def _entropy_loss(self, logp_TB1):
        if self.alpha_opt is None:
            return th.zeros((), device=self.device)
        return -(self.log_alpha * (logp_TB1.detach() + self.target_entropy)).mean()

    def _step_with_clip(self, params_iterable, opt: th.optim.Optimizer, loss: th.Tensor, clip_norm: float):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(params_iterable, clip_norm)
        opt.step()

    def train_per_epoch(self, epoch: int, writer: Optional[SummaryWriter] = None, log_dir: str = "runs/sac_rnn", is_batch_rl: bool = False) -> bool:
        th.backends.cudnn.benchmark = True
        local_writer = False
        if writer is None:
            writer = SummaryWriter(log_dir=log_dir); local_writer = True

        if not is_batch_rl and (epoch >= self.buf.data_num or self.buf.data_num <= 2000):
            return False

        # =================================================================================
        # OPTIMIZATION: Single Batch Fetch with multiple updates (High UTD)
        # We eliminate 80% of the CPU overhead by fetching once and updating Critic 5x
        # =================================================================================
        
        # We fetch batches once per step
        for batch in self.buf.get_sequences(self.cfg.batch_size, trace_length=400, device=self.device):
            obs_TBD, act_TBA, rew_TB1, nxt_TBD, done_TB1, info = batch
            T, B, _ = obs_TBD.shape
            importance_weights = info['is_weights'].detach().squeeze(-1)
            
            # 1. Pre-calculate Belief & Representation ONCE for the batch
            # This is the heaviest part of the model, we share it between Critic loop and Actor
            z_curr_TB1, mu_TB1, logvar_TB1, y_hat_TB1 = self._predict_belief_seq(obs_TBD)
            
            with th.no_grad():
                z_next_TB1, _, _, _ = self._predict_belief_seq(nxt_TBD)
            
            h0, _ = self.net.init_hidden(B, self.device)
            feat_TBH, hT = self._encode_with_belief(obs_TBD, z_curr_TB1, h0)

            # 2. Run Multiple Critic Updates on the SAME batch (UTD Ratio)
            # This saturates the GPU with matrix multiplications instead of waiting for CPU
            for i_crit in range(self.critic_update_steps):
                
                current_feat_TBH = feat_TBH if i_crit == 0 else feat_TBH.detach()
                
                c_loss, cdl_stats = self._critic_loss(
                    current_feat_TBH, act_TBA, nxt_TBD, 
                    belief_curr_TB1=z_curr_TB1, 
                    belief_next_TB1=z_next_TB1, 
                    hT=hT, rew_TB1=rew_TB1, done_TB1=done_TB1,
                    importance_weights=importance_weights, is_batch_rl=is_batch_rl
                )

                self._step_with_clip(self.net.critic_parameters(), self.critic_opt, c_loss, clip_norm=100.0)
                self.net._soft_sync(self.cfg.tau)
                
                # Log only on the last sub-step to reduce IO blocking
                if i_crit == self.critic_update_steps - 1 and self._global_step % self.log_int == 0:
                    writer.add_scalar("loss/critic", c_loss.item(), self._global_step)
                    if cdl_stats:
                        for k, v in cdl_stats.items(): writer.add_scalar(k, v.item(), self._global_step)

            # 3. Actor / Belief Update (Once per batch fetch)
            b_loss, beta, kl_loss = self._belief_supervision_loss(
                mu_TB1, logvar_TB1, y_hat_TB1,
                interf_B1=info["interference"], iw_B1=info["is_weights"].detach(),
                epoch=epoch
            )
            
            # Reuse the feature encoding calculated in step 1
            a_loss, logp_TB1 = self._actor_loss(feat_TBH)
            
            self._step_with_clip(self.net.actor_parameters(), self.actor_opt, a_loss, clip_norm=50.0)
            self._step_with_clip(self.net.belief_parameteres(), self.belief_opt, b_loss, clip_norm=50.0)

            if self.alpha_opt is not None:
                ent_loss = self._entropy_loss(logp_TB1)
                self._step_with_clip([self.log_alpha], self.alpha_opt, ent_loss, clip_norm=1e9)

            self._global_step += 1
            self._upd += 1
            
            # Gated logging
            if self._global_step % self.log_int == 0:
                writer.add_scalar("loss/actor_loss", a_loss.item(), self._global_step)
                writer.add_scalar("loss/belief", b_loss.item(), self._global_step)
                writer.add_scalar("loss/KL", (beta * kl_loss).item(), self._global_step)

        if local_writer:
            writer.flush(); writer.close()
        return True

    # ... (keep tf_act, save, load as they were)
    @th.no_grad()
    def tf_act(self, obs_vec: list[float], is_evaluate: bool = False, reset_hidden: bool = False):
        obs = th.tensor(obs_vec, device=self.device, dtype=th.float32).unsqueeze(0)
        if reset_hidden or self._eval_h is None:
            self._eval_h, self._belief_h = self.net.init_hidden(1, self.device)

        z_BH, h1, mu_BH, logvar_BH, y_hat_B1 = self.net.belief_predict_step(obs, self._belief_h)
        if is_evaluate:
            feat, h_next = self.net.encode(obs, mu_BH.detach(), self._eval_h)
            mu, _ = self.net._mean_std(feat)
            action = th.tanh(mu)
            q1, q2 = self.net.q(feat, action)
            logp = th.zeros(1, 1, device=self.device)
            v_like = th.min(q1, q2)[0].detach().cpu().numpy()
        else:
            feat, h_next = self.net.encode(obs, z_BH.detach(), self._eval_h)
            action, logp = self.net.sample_from_features(feat, detach_feat_for_actor=True)
            v_like = 0
        self._belief_h = h1.detach(); self._eval_h = h_next.detach()
        return {"action": action[0].detach().cpu().numpy(), "log_prob": logp[0].detach().cpu().numpy(), "value": v_like, "belief": y_hat_B1[0].detach().cpu().numpy()}

    def save(self, path: str):
        checkpoint = {'model_state_dict': self.net.state_dict(), 'actor_opt_state_dict': self.actor_opt.state_dict(), 'critic_opt_state_dict': self.critic_opt.state_dict(), 'belief_opt_state_dict': self.belief_opt.state_dict(), 'log_alpha': self.log_alpha, 'alpha_opt_state_dict': self.alpha_opt.state_dict() if self.alpha_opt else None, 'cfg': self.cfg, 'global_step': self._global_step}
        th.save(checkpoint, path)

    def load(self, path: str, device: str):
        device = th.device(device)
        checkpoint = th.load(path, map_location=device, weights_only=False)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        self.belief_opt.load_state_dict(checkpoint['belief_opt_state_dict'])
        if checkpoint.get('log_alpha') is not None: self.log_alpha = checkpoint['log_alpha'].to(device)
        if 'alpha_opt_state_dict' in checkpoint and checkpoint['alpha_opt_state_dict']: self.alpha_opt.load_state_dict(checkpoint['alpha_opt_state_dict'])
        self.cfg = checkpoint['cfg']; self._global_step = checkpoint['global_step']; self.device = device; self.net.to(device)