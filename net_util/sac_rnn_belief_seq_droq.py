from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import importlib
import copy

from net_util.base import PolicyBase
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
    batch_size: int = 256
    learning_starts: int = 2_000
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    target_update_interval: int = 1
    ent_coef: str | float = "auto"     # "auto" or float
    ent_coef_init: float = 1.0
    target_entropy: str | float = "auto"  # "auto" -> -act_dim
    act_dim: int = 2
    device: str = "cpu"

    # fields referenced below
    seed: int = 0
    obs_dim: int = 6
    belief_dim: int = 1
    network_module_path: str = "net_util.model.sac_rnn_belief"
    load_path = "latest.pt"
    
    batch_rl = False
    
    # --- retraining timer (0 disables) ---
    retrain_interval: int = 0                 # e.g., 500; 0 => off
    retrain_reset_alpha: bool = True          # reset temperature (α) when retraining
    retrain_reset_targets: bool = True        # reinit target nets from freshly reset critics

@register_policy
class SACRNNBeliefSeqDroq(PolicyBase):
    """Replay buffer must yield (obs, act, rew, next_obs, done) minibatches."""
    def __init__(self, cmd_cls, cfg: SACRNNBeliefSeqDroq_Config, rollout_buffer: RNNPriReplayBuffer2 | None = None, device: str | None = None, state_transform_dict = None):
        super().__init__(cmd_cls, state_transform_dict = state_transform_dict)
        th.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.cfg = cfg
        self.device = th.device(cfg.device) if device is None else th.device(device)

        # Load network module
        net_mod = importlib.import_module(cfg.network_module_path)
        Network = getattr(net_mod, "Network")
        self.net = Network(cfg.obs_dim, cfg.act_dim, scale_log_offset=0, belief_dim = cfg.belief_dim).to(self.device) # type: ignore
        self.buf = rollout_buffer

        # optimizers
        ac_lr = cfg.lr * 0.1 if cfg.batch_rl else cfg.lr
        self.actor_opt  = th.optim.Adam(self.net.actor_parameters(),  lr=ac_lr)
        self.critic_opt = th.optim.Adam(self.net.critic_parameters(), lr=cfg.lr)
        self.belief_opt = th.optim.Adam(self.net.belief_parameteres(), lr=cfg.lr)

        # # temperature α
        if cfg.ent_coef == "auto" and cfg.batch_rl == False:
            self.log_alpha = th.tensor([cfg.ent_coef_init], device=self.device).log().requires_grad_(True)
            self.alpha_opt = th.optim.Adam([self.log_alpha], lr=cfg.lr)
            self.target_entropy = -float(cfg.act_dim) if cfg.target_entropy == "auto" else float(cfg.target_entropy)
        else:
            self.log_alpha, self.alpha_opt = None, None
            self.alpha_tensor = th.tensor(float(0.2), device=self.device)
        
        # self.log_alpha, self.alpha_opt = None, None
        # self.alpha_tensor = th.tensor(float(0.2), device=self.device)

        # counters / state
        self._upd = 0
        self._epoch_h = None
        self._eval_h = None
        self._belief_h = None
        self._global_step = 0
        
        self.log_int = 50
        
        self.last_obs = None
        
        self.annealing_bl = lambda epoch, max_lb: min(epoch / 50e3 * max_lb, max_lb)
        
        self.retrain_idx = 0
        self.max_retrain = 5
        
        self.critic_update_steps = 10

    def _alpha(self):
        return (self.log_alpha.exp() if self.log_alpha is not None else self.alpha_tensor).detach()

    def _reset_modules_excluding_belief_(self):
        """
        Reinitialize all submodules that are *not* part of the belief network.
        Uses each module's reset_parameters() if present; leaves belief as-is.
        """
        # Build a set of Param object ids that belong to belief
        belief_param_ids = {id(p) for p in self.net.belief_parameteres()}

        for m in self.net.modules():
            # only operate at this module's *own* params (avoid double-initting via recurse)
            own_params = list(m.parameters(recurse=False))
            if not own_params:
                continue
            # If any param in this module belongs to belief, skip the whole module
            if any(id(p) in belief_param_ids for p in own_params):
                continue
            # Otherwise, if the module knows how to reset itself, do it
            if hasattr(m, "reset_parameters") and callable(getattr(m, "reset_parameters")):
                m.reset_parameters()

    @th.no_grad()
    def _hard_copy_(self, tgt: nn.Module, src: nn.Module):
        for tp, sp in zip(tgt.parameters(), src.parameters()):
            tp.data.copy_(sp.data)


    def cdl_loss(
        self,
        feat_TBH: th.Tensor,                 # [T,B,H]
        nxt_TBD: th.Tensor,                  # [T,B,D]
        belief_seq_TB1_sac: th.Tensor,       # [T,B,1]
        act_TBA: th.Tensor,                  # [T,B,A]
        num_random: int = 10,
        beta_cql: float = 0.05,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        device = feat_TBH.device
        T, B, H = feat_TBH.shape
        _, _, A = act_TBA.shape

        with th.no_grad():
            # random actions at s_t
            a_rand_TBNA = th.rand(T, B, num_random, A, device=device) * 2 - 1

            # π(s_t) actions
            a_curr_TBA, _ = self.net.sample_from_features(feat_TBH, detach_feat_for_actor=True)
            a_curr_TBNA = a_curr_TBA.unsqueeze(2).expand(T, B, num_random, A)

            # π(s_{t+1}) actions
            h0_enc, _ = self.net.init_hidden(B, device)
            feat_next_TBH, _ = self.net.encode_seq(nxt_TBD, belief_seq_TB1_sac, h0_enc)
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

        q1_bc, q2_bc = self.net.q(feat_TBH, act_TBA)        # [T,B,1]
        q1_bc_TB11 = q1_bc.unsqueeze(2)                     # [T,B,1,1]
        q2_bc_TB11 = q2_bc.unsqueeze(2)

        cat_q1 = th.cat([q1_rand, q1_bc_TB11, q1_next, q1_curr], dim=2)  # [T,B,Ntot,1]
        cat_q2 = th.cat([q2_rand, q2_bc_TB11, q2_next, q2_curr], dim=2)

        # log-sum-exp over proposals
        lse1_TB = th.logsumexp(cat_q1.squeeze(-1), dim=2)   # [T,B]
        lse2_TB = th.logsumexp(cat_q2.squeeze(-1), dim=2)

        # penalties (anchor by data action)
        cql1 = (lse1_TB.mean() - q1_bc.mean())
        cql2 = (lse2_TB.mean() - q2_bc.mean())
        penalty = beta_cql * (cql1 + cql2)

        # stats for logging
        min_q_rand = th.minimum(q1_rand, q2_rand).mean()
        min_q_curr = th.minimum(q1_curr, q2_curr).mean()
        min_q_next = th.minimum(q1_next, q2_next).mean()
        min_q_bc   = th.minimum(q1_bc,   q2_bc).mean()
        
        stats = {
            "cdl/penalty_total": penalty.detach(),
            # "cdl/cql1": cql1.detach(),
            # "cdl/cql2": cql2.detach(),
            # "cdl/minQ_bc_mean":   min_q_bc.detach(),
            # "cdl/minQ_rand_mean": min_q_rand.detach(),
            # "cdl/minQ_curr_mean": min_q_curr.detach(),
            # "cdl/minQ_next_mean": min_q_next.detach(),
            "cdl/gap_rand_bc": (min_q_rand - min_q_bc).detach(),
            "cdl/gap_curr_bc": (min_q_curr - min_q_bc).detach(),
            # "cdl/gap_next_bc": (min_q_next - min_q_bc).detach(),
            # "cdl/std_cat_q1": cat_q1.squeeze(-1).std(dim=2).mean().detach(),
            # "cdl/std_cat_q2": cat_q2.squeeze(-1).std(dim=2).mean().detach(),
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
        beta     = self.annealing_bl(epoch, 10)
        return mse_loss + beta * kl_loss, beta, kl_loss

    def _critic_loss(self, feat_TBH, act_TBA, nxt_TBD, belief_TB1, hT, rew_TB1, done_TB1, importance_weights, is_batch_rl: bool):
        alpha = self._alpha()
        with th.no_grad():
            backup_TB1 = self.net.target_backup_seq(nxt_TBD, belief_TB1, hT, rew_TB1, done_TB1, self.cfg.gamma, alpha)

        q1_TB1, q2_TB1 = self.net.q(feat_TBH, act_TBA)
        diff = th.stack([q1_TB1, q2_TB1], dim=0) - backup_TB1
        diff = diff / self.buf.sigma
        c_loss_per_t = diff.pow(2).mean(dim=0)      # [T,B,1]
        c_loss_batch = c_loss_per_t.mean(dim=(0,2)) # [B]
        c_loss = (c_loss_batch * importance_weights).mean()

        cdl_stats = None
        if is_batch_rl:
            cdl_penalty, cdl_stats = self.cdl_loss(
                feat_TBH=feat_TBH, nxt_TBD=nxt_TBD, belief_seq_TB1_sac=belief_TB1,
                act_TBA=act_TBA, num_random=10, beta_cql=5 * 20 / self.buf.sigma,
            )
            c_loss = c_loss + cdl_penalty
        return c_loss, cdl_stats

    def _actor_loss(self, feat_TBH):
        with self.net.critics_frozen():
            a_pi_TBA, logp_TB1 = self.net.sample_from_features(feat_TBH)
            q1_pi, q2_pi = self.net.q(feat_TBH.detach(), a_pi_TBA)
            alpha = self._alpha()
            a_loss = (alpha * logp_TB1 - (q1_pi + q2_pi) / 2).mean()
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

        # (optional) early return gate
        if not is_batch_rl and (epoch >= self.buf.data_num or self.buf.data_num <= 10000):
            return False

        # ---------- Critic + Belief phase ----------
        for _ in range(self.critic_update_steps):
            for batch in self.buf.get_sequences(self.cfg.batch_size, trace_length=100, device=self.device):
                obs_TBD, act_TBA, rew_TB1, nxt_TBD, done_TB1, info = batch
                T, B, _ = obs_TBD.shape
                importance_weights = info['is_weights'].detach().squeeze(-1)

                h0, _ = self.net.init_hidden(B, self.device)
                z_TB1, mu_TB1, logvar_TB1, y_hat_TB1 = self._predict_belief_seq(obs_TBD)
                b_loss, beta, kl_loss = self._belief_supervision_loss(
                    mu_TB1, logvar_TB1, y_hat_TB1,
                    interf_B1=info["interference"], iw_B1=info["is_weights"].detach(),
                    epoch=epoch
                )

                feat_TBH, hT = self._encode_with_belief(obs_TBD, z_TB1, h0)
                c_loss, cdl_stats = self._critic_loss(
                    feat_TBH, act_TBA, nxt_TBD, z_TB1, hT, rew_TB1, done_TB1,
                    importance_weights, is_batch_rl
                )

                # optimize
                self._step_with_clip(self.net.critic_parameters(), self.critic_opt, c_loss, clip_norm=10.0)
                self._step_with_clip(self.net.belief_parameteres(), self.belief_opt, b_loss, clip_norm=5.0)

                # targets
                self.net._soft_sync(self.cfg.tau)
                self._upd += 1
                self._global_step += 1

                if cdl_stats is not None:
                    for k, v in cdl_stats.items():
                        writer.add_scalar(k, v.item(), self._global_step)
                
                writer.add_scalar("loss/critic", c_loss.item(), self._global_step)
                writer.add_scalar("loss/belief", b_loss.item(), self._global_step)
                writer.add_scalar("loss/KL", (beta * kl_loss).item(), self._global_step)


        # ---------- Actor (+ temperature) phase ----------
        for batch in self.buf.get_sequences(self.cfg.batch_size, trace_length=100, device=self.device):
            obs_TBD, act_TBA, rew_TB1, nxt_TBD, done_TB1, info = batch
            T, B, _ = obs_TBD.shape

            h0, _ = self.net.init_hidden(B, self.device)
            z_TB1, mu_TB1, logvar_TB1, y_hat_TB1 = self._predict_belief_seq(obs_TBD)
            feat_TBH, hT = self._encode_with_belief(obs_TBD, z_TB1, h0)

            a_loss, logp_TB1 = self._actor_loss(feat_TBH)
            self._step_with_clip(self.net.actor_parameters(), self.actor_opt, a_loss, clip_norm=5.0)

            if self.alpha_opt is not None:
                ent_loss = self._entropy_loss(logp_TB1)
                self._step_with_clip([self.log_alpha], self.alpha_opt, ent_loss, clip_norm=1e9)  # no real grad clipping needed

            # ---------- Epoch-end logging (kept minimal/clean) ----------
            writer.add_scalar("loss/actor_loss", a_loss.item(), epoch)
            if self.alpha_opt is not None:
                writer.add_scalar("loss/ent_loss", ent_loss.item(), epoch)

        if local_writer:
            writer.flush(); writer.close()
        return True


    @th.no_grad()
    def tf_act(self, obs_vec: list[float], is_evaluate: bool = False, reset_hidden: bool = False):
        """
        Inference for a single observation (obs_vec is a Python list of length D).
        Returns:
        {
            "action":   np.ndarray (act_dim,),
            "log_prob": np.ndarray (1,),
            "value":    np.ndarray (1,),   # min(Q1,Q2)(s,a)
        }
        """
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

        self._belief_h = h1.detach()
        self._eval_h = h_next.detach()
        

        return {
            "action":   action[0].detach().cpu().numpy(),
            "log_prob": logp[0].detach().cpu().numpy(),
            "value":    v_like,
            "belief":   y_hat_B1[0].detach().cpu().numpy(),
        }
        
    def save(self, path: str):
        """Save the model and optimizer states."""
        checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'actor_opt_state_dict': self.actor_opt.state_dict(),
            'critic_opt_state_dict': self.critic_opt.state_dict(),
            'belief_opt_state_dict': self.belief_opt.state_dict(),
            'log_alpha': self.log_alpha,  # Save the tensor itself
            'alpha_opt_state_dict': self.alpha_opt.state_dict() if self.alpha_opt else self.alpha_opt,
            'cfg': self.cfg,
            'global_step': self._global_step,
        }
        th.save(checkpoint, path)


    def load(self, path: str, device: str):
        """Load the model and optimizer states."""
        device = th.device(device)
        checkpoint = th.load(path, map_location=device, weights_only=False)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        self.belief_opt.load_state_dict(checkpoint['belief_opt_state_dict'])
        
        if checkpoint.get('log_alpha') is not None:
            self.log_alpha = checkpoint['log_alpha'].to(device)  # Directly assign the tensor

        if 'alpha_opt_state_dict' in checkpoint and checkpoint['alpha_opt_state_dict']:
            self.alpha_opt.load_state_dict(checkpoint['alpha_opt_state_dict'])
        
        self.cfg = checkpoint['cfg']
        self._global_step = checkpoint['global_step']
        self.device = device
        self.net.to(device)

