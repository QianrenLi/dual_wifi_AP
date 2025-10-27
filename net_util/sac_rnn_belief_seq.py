from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import importlib
import copy

from net_util.base import PolicyBase
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
class SACRNNBeliefSeq_Config:
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

@register_policy
class SACRNNBeliefSeq(PolicyBase):
    """Replay buffer must yield (obs, act, rew, next_obs, done) minibatches."""
    def __init__(self, cmd_cls, cfg: SACRNNBeliefSeq_Config, rollout_buffer: RNNPriReplayBuffer2 | None = None, device: str | None = None, state_transform_dict = None):
        super().__init__(cmd_cls, state_transform_dict = state_transform_dict)
        th.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.cfg = cfg
        self.device = th.device(cfg.device) if device is None else th.device(device)

        # Load network module
        net_mod = importlib.import_module(cfg.network_module_path)
        Network = getattr(net_mod, "Network")
        self.net = Network(cfg.obs_dim, cfg.act_dim, scale_log_offset=0, belief_dim = cfg.belief_dim).to(self.device)
        self.buf = rollout_buffer

        # optimizers
        self.actor_opt  = th.optim.Adam(self.net.actor_parameters(),  lr=cfg.lr)
        self.critic_opt = th.optim.Adam(self.net.critic_parameters(), lr=cfg.lr)
        self.belief_opt = th.optim.Adam(self.net.belief_parameteres(), lr=cfg.lr)

        # temperature α
        if cfg.ent_coef == "auto":
            self.log_alpha = th.tensor([cfg.ent_coef_init], device=self.device).log().requires_grad_(True)
            self.alpha_opt = th.optim.Adam([self.log_alpha], lr=cfg.lr)
            self.target_entropy = -float(cfg.act_dim) if cfg.target_entropy == "auto" else float(cfg.target_entropy)
        else:
            self.log_alpha, self.alpha_opt = None, None
            self.alpha_tensor = th.tensor(float(cfg.ent_coef), device=self.device)

        # counters / state
        self._upd = 0
        self._epoch_h = None
        self._eval_h = None
        self._belief_h = None
        self._belief = None
        self._global_step = 0
        
        self.log_int = 50
        
        self.last_obs = None

    def _alpha(self):
        return (self.log_alpha.exp() if self.log_alpha is not None else self.alpha_tensor).detach()


    def train_per_epoch(self, epoch: int, writer: Optional[SummaryWriter] = None, log_dir: str = "runs/sac_rnn") -> bool:
        th.backends.cudnn.benchmark = True
        use_cuda_amp = False
        scaler = th.amp.GradScaler('cuda', enabled=use_cuda_amp)

        local_writer = False
        if writer is None:
            writer = SummaryWriter(log_dir=log_dir); local_writer = True

        actor_losses, critic_losses, ent_losses = [], [], []
        q1_means, q2_means, qmin_pi_means, logp_means = [], [], [], []

        # One pass per sequence batch
        got_any = False
        for batch in self.buf.get_sequences(self.cfg.batch_size, trace_length=100, device=self.device):
            got_any = True
            obs_TBD, act_TBA, rew_TB1, nxt_TBD, done_TB1, info = batch  # time-major
            T, B, _ = obs_TBD.shape

            # Init hiddens for this batch
            h0, b0 = self.net.init_hidden(B, self.device)
            if self._belief is None or self._belief.shape[0] != B:
                self._belief = th.zeros(B, self.cfg.belief_dim, device=self.device)
            if self._belief_h is None or self._belief_h.shape[1] != B:
                self._belief_h = b0.clone()

            # ---- Belief sequence ----
            belief_seq_TB1, bT = self.net.belief_predict_seq(obs_TBD, self._belief_h)  # [T,B,1]
            # simple supervision toward per-episode target in info["interference"] (B,1)
            
            interf_B1 = info["interference"]                     # [B,1]
            belief_mean_B1 = belief_seq_TB1.mean(dim=0)          # [B,1]
            resid2 = (interf_B1 - belief_mean_B1).pow(2)
            const = th.log(th.tensor(2.0 * np.pi, device=self.device, dtype=th.float32))
            b_loss = 0.5 * (resid2 + const).mean()
            
            belief_seq_TB1_sac = belief_seq_TB1.detach()

            # ---- Encode whole sequence (online) ----
            with th.autocast(device_type='cuda', enabled=use_cuda_amp):
                feat_TBH, hT = self.net.encode_seq(obs_TBD, belief_seq_TB1_sac, h0)  # use detached belief
                a_pi_TBA, logp_TB1 = self.net.sample_from_features(feat_TBH)

                if self.alpha_opt is not None:
                    ent_loss = -(self.log_alpha * (logp_TB1.detach() + self.target_entropy)).mean()
                else:
                    ent_loss = th.zeros((), device=self.device)
                alpha = self._alpha()

                with th.no_grad():
                    backup_TB1 = self.net.target_backup_seq(
                        nxt_TBD, belief_seq_TB1_sac, hT, rew_TB1, done_TB1, self.cfg.gamma, alpha
                    )
                q1_TB1, q2_TB1 = self.net.q(feat_TBH, act_TBA)
                diff = th.stack([q1_TB1, q2_TB1], dim=0) - backup_TB1
                diff = diff / self.buf.sigma
                c_loss_per_t = diff.pow(2).mean(dim=0)          # [T,B,1]
                c_loss_batch = c_loss_per_t.mean(dim=(0,2))     # [B]
                c_loss = c_loss_batch.mean()

            # ---- Optimize (same order is fine now; graphs disjoint) ----
            self.critic_opt.zero_grad(set_to_none=True)
            scaler.scale(c_loss).backward()
            critic_gn = th.nn.utils.clip_grad_norm_(self.net.critic_parameters(), 10.0)
            scaler.step(self.critic_opt)

            self.actor_opt.zero_grad(set_to_none=True)
            with self.net.critics_frozen():
                with th.autocast(device_type='cuda', enabled=use_cuda_amp):
                    q1_pi, q2_pi = self.net.q(feat_TBH.detach(), a_pi_TBA)
                    qmin_pi = th.min(q1_pi, q2_pi)
                    a_loss = (alpha * logp_TB1 - qmin_pi).mean()
            scaler.scale(a_loss).backward()
            actor_gn = th.nn.utils.clip_grad_norm_(self.net.actor_parameters(), 5.0)
            scaler.step(self.actor_opt)

            if self.alpha_opt is not None:
                self.alpha_opt.zero_grad(set_to_none=True)
                scaler.scale(ent_loss).backward()
                scaler.step(self.alpha_opt)

            self.belief_opt.zero_grad(set_to_none=True)
            scaler.scale(b_loss).backward()
            b_gn = th.nn.utils.clip_grad_norm_(self.net.belief_parameteres(), 5.0)
            scaler.step(self.belief_opt)

            scaler.update()

            # Soft updates
            if (self._upd % self.cfg.target_update_interval) == 0:
                tau = self.cfg.tau
                with th.no_grad():
                    for tp, sp in zip(self.net.q1_t.parameters(), self.net.q1.parameters()):
                        tp.mul_(1 - tau).add_(sp, alpha=tau)
                    for tp, sp in zip(self.net.q2_t.parameters(), self.net.q2.parameters()):
                        tp.mul_(1 - tau).add_(sp, alpha=tau)
                    for tp, sp in zip(self.net.fe_t.parameters(), self.net.fe.parameters()):
                        tp.mul_(1 - tau).add_(sp, alpha=tau)
            self._upd += 1

            # aggregates
            actor_losses.append(a_loss.item())
            critic_losses.append(c_loss.item())
            q1_means.append(q1_TB1.mean().item())
            q2_means.append(q2_TB1.mean().item())
            qmin_pi_means.append(qmin_pi.mean().item())
            logp_means.append(logp_TB1.mean().item())
            if self.alpha_opt is not None: ent_losses.append(ent_loss.item())

            # update priorities using per-episode loss
            self.buf.update_episode_losses(info["ep_ids"], c_loss_batch.detach().cpu().numpy())

            # carry belief hidden for next batch
            self._belief_h = self.net.belief_rnn.init_state(B, self.device)
            self._belief   = th.zeros(B, self.cfg.belief_dim, device=self.device)

            self._global_step += 1

        if not got_any:
            return False

        # epoch-end aggregates
        writer.add_scalar("loss/actor_epoch",  _safe_mean(actor_losses), epoch)
        writer.add_scalar("loss/critic_epoch", _safe_mean(critic_losses), epoch)
        writer.add_scalar("policy/logp_pi_epoch", _safe_mean(logp_means), epoch)
        writer.add_scalar("q/q1_mean_epoch", _safe_mean(q1_means), epoch)
        writer.add_scalar("q/q2_mean_epoch", _safe_mean(q2_means), epoch)
        writer.add_scalar("q/qmin_pi_epoch", _safe_mean(qmin_pi_means), epoch)
        if ent_losses: writer.add_scalar("loss/entropy_epoch", _safe_mean(ent_losses), epoch)

        if local_writer:
            writer.flush(); writer.close()

        # reset carried states for next epoch if you prefer
        self._belief_h = None
        self._belief = None
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
            self._belief = th.zeros(1, self.cfg.belief_dim, device = self.device)

        feat, h_next = self.net.encode(obs, self._belief, self._eval_h)

        if is_evaluate:
            mu, _ = self.net._mean_std(feat)
            action = th.tanh(mu)
            q1, q2 = self.net.q(feat, action)
            logp = th.zeros(1, 1, device=self.device)
            v_like = th.min(q1, q2)[0].detach().cpu().numpy()
        else:
            action, logp = self.net.sample_from_features(feat, detach_feat_for_actor=True)
            v_like = 0

        self._eval_h = h_next.detach()
        
        self._belief, self._belief_h = self.net.belief_predict(obs, self._belief_h)
        self._belief = self._belief.detach()
        self._belief_h = self._belief_h.detach()

        return {
            "action":   action[0].detach().cpu().numpy(),
            "log_prob": logp[0].detach().cpu().numpy(),
            "value":    v_like
        }
        
    def save(self, path: str):
        """Save the model and optimizer states."""
        checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'actor_opt_state_dict': self.actor_opt.state_dict(),
            'critic_opt_state_dict': self.critic_opt.state_dict(),
            'belief_opt_state_dict': self.belief_opt.state_dict(),
            'log_alpha': self.log_alpha,  # Save the tensor itself
            'alpha_opt_state_dict': self.alpha_opt.state_dict(),
            'cfg': self.cfg,
            'global_step': self._global_step,
        }
        th.save(checkpoint, path)


    def load(self, path: str, device: str):
        """Load the model and optimizer states."""
        checkpoint = th.load(path, map_location=device, weights_only=False)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        self.belief_opt.load_state_dict(checkpoint['belief_opt_state_dict'])
        
        self.log_alpha = checkpoint['log_alpha'].to(device)  # Directly assign the tensor
        self.alpha_opt.load_state_dict(checkpoint['alpha_opt_state_dict'])
        
        self.cfg = checkpoint['cfg']
        self._global_step = checkpoint['global_step']
        self.device = device
        self.net.to(device)

