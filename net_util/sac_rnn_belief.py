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
from util.trace_collec import flatten_leaves, shift_res_action_in_states
from . import register_policy, register_policy_cfg
from torch.utils.tensorboard import SummaryWriter

# ---------------- utils ----------------
def _safe_mean(xs):
    return float(np.mean(xs)) if xs else float("nan")

def symlog(x: th.Tensor, eps=1e-12) -> th.Tensor:
    return th.sign(x) * th.log(th.abs(x) + 1.0 + eps)

def _grad_global_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.data.pow(2).sum().item())
    return float(total ** 0.5)

@register_policy_cfg
@dataclass
class SACRNNPri_Config:
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
    network_module_path: str = "net_util.model.sac_rnn_belief"

@register_policy
class SACRNNPri(PolicyBase):
    """Replay buffer must yield (obs, act, rew, next_obs, done) minibatches."""
    def __init__(self, cmd_cls, cfg: SACRNNPri_Config, rollout_buffer: RNNPriReplayBuffer2 | None = None, device: str | None = None, state_transform_dict = None):
        super().__init__(cmd_cls, state_transform_dict = state_transform_dict)
        th.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.cfg = cfg
        self.device = th.device(cfg.device) if device is None else th.device(device)

        # Load network module
        net_mod = importlib.import_module(cfg.network_module_path)
        Network = getattr(net_mod, "Network")
        self.net = Network(cfg.obs_dim, cfg.act_dim, scale_log_offset=self.cmd_cls.sum_log_scales(), rnn_k = cfg.rnn_dim).to(self.device)
        self.buf = rollout_buffer

        # optimizers
        self.actor_opt  = th.optim.Adam(self.net.actor_parameters(),  lr=cfg.lr)
        self.critic_opt = th.optim.Adam(self.net.critic_parameters(), lr=cfg.lr)

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
        self._global_step = 0
        
        self.last_obs = None

    def _alpha(self):
        return (self.log_alpha.exp() if self.log_alpha is not None else self.alpha_tensor).detach()

    def train_per_epoch(self, epoch: int, writer: Optional[SummaryWriter] = None, log_dir: str = "runs/sac_rnn") -> bool:
        """Run one epoch. Logs per-minibatch metrics to TensorBoard."""
        # if getattr(self.buf, "ptr", 0) < max(self.cfg.learning_starts, self.cfg.batch_size):
        #     return False

        local_writer = False
        if writer is None:
            writer = SummaryWriter(log_dir=log_dir)
            local_writer = True

        # buffer stats
        writer.add_scalar("buffer/ptr", getattr(self.buf, "ptr", 0), epoch)
        writer.add_scalar("buffer/size", getattr(self.buf, "size", 0), epoch)

        # collect epoch aggregates
        actor_losses, critic_losses, ent_losses = [], [], []
        q1_means, q2_means, qmin_pi_means, logp_means = [], [], [], []

        steps, self._epoch_h = 0, None
        for obs, act, rew, nxt, done, info in self.buf.get_minibatches(self.cfg.batch_size):
            obs  = obs.to(self.device).float()
            act  = act.to(self.device).float()
            rew  = rew.to(self.device).float().view(-1, 1)
            nxt  = nxt.to(self.device).float()
            done = done.to(self.device).float().view(-1, 1)
            
            if self._epoch_h is None:
                self._epoch_h = self.net.init_hidden(obs.size(0), self.device)
            
            if self._epoch_h.size(0) != obs.size(0):
                break
            
            # encode once
            feat_t, h_tp1 = self.net.encode(obs, self._epoch_h)

            # one policy sample reused (α + actor)
            a_pi, logp_pi = self.net.sample_from_features(feat_t, detach_feat_for_actor=True)
            # α step (α only)
            if self.alpha_opt is not None:
                ent_loss = -(self.log_alpha * (logp_pi.detach() + self.target_entropy)).mean()
                self.alpha_opt.zero_grad(); ent_loss.backward(); self.alpha_opt.step()
                writer.add_scalar("sac/alpha", self._alpha().item(), self._global_step)
                writer.add_scalar("loss/entropy_step", ent_loss.item(), self._global_step)
                ent_losses.append(ent_loss.item())
            alpha = self._alpha()

            # targets at (nxt, h_{t+1})
            with th.no_grad():
                backup = self.net.target_backup(nxt, h_tp1, rew, done, self.cfg.gamma, alpha)

            # critic step: FE + critics
            self.critic_opt.zero_grad()
            q1_pred, q2_pred = self.net.q(feat_t, act)
            
            diff = (th.stack([q1_pred, q2_pred], 0) - backup) / self.buf.sigma      # [2, B, ...]
            c_loss_batch = diff.pow(2).mean(dim=(0, 2))
            
            c_loss = c_loss_batch.mean()
            c_loss.backward()

            # critic grad stats (before step)
            critic_gn = th.nn.utils.clip_grad_norm_(self.net.critic_parameters(), 10.0)
            self.critic_opt.step()

            # actor step: heads only; critics frozen; FE detached
            self.actor_opt.zero_grad()
            with self.net.critics_frozen():
                q1_pi, q2_pi = self.net.q(feat_t.detach(), a_pi)
                qmin_pi = th.min(q1_pi, q2_pi)
                a_loss = (alpha * logp_pi - qmin_pi).mean()
                a_loss.backward()

                # actor grad stats (before step)
                actor_gn = th.nn.utils.clip_grad_norm_(self.net.actor_parameters(), 5.0)
                self.actor_opt.step()

            # carry hidden; target soft updates
            self._epoch_h = h_tp1.detach()
            if (self._upd % self.cfg.target_update_interval) == 0:
                for tp, sp in zip(self.net.q1_t.parameters(), self.net.q1.parameters()):
                    tp.data.mul_(1 - self.cfg.tau).add_(sp.data, alpha=self.cfg.tau)
                for tp, sp in zip(self.net.q2_t.parameters(), self.net.q2.parameters()):
                    tp.data.mul_(1 - self.cfg.tau).add_(sp.data, alpha=self.cfg.tau)
            self._upd += 1

            # ---------- per-batch logging ----------
            writer.add_scalar("loss/critic_step", c_loss.item(), self._global_step)
            writer.add_scalar("loss/actor_step",  a_loss.item(), self._global_step)
            writer.add_scalar("policy/logp_pi_step", logp_pi.mean().item(), self._global_step)
            writer.add_scalar("q/q1_pred_mean_step", q1_pred.mean().item(), self._global_step)
            writer.add_scalar("q/q2_pred_mean_step", q2_pred.mean().item(), self._global_step)
            writer.add_scalar("q/qmin_pi_step", qmin_pi.mean().item(), self._global_step)
            writer.add_scalar("grad/critic_global_norm", critic_gn, self._global_step)
            writer.add_scalar("grad/actor_global_norm",  actor_gn,  self._global_step)
            # LRs
            writer.add_scalar("opt/critic_lr", self.critic_opt.param_groups[0]["lr"], self._global_step)
            writer.add_scalar("opt/actor_lr",  self.actor_opt.param_groups[0]["lr"],  self._global_step)

            # aggregates
            actor_losses.append(a_loss.item())
            critic_losses.append(c_loss.item())
            q1_means.append(q1_pred.mean().item())
            q2_means.append(q2_pred.mean().item())
            qmin_pi_means.append(qmin_pi.mean().item())
            logp_means.append(logp_pi.mean().item())

            self.buf.update_episode_losses(info["ep_ids"], c_loss_batch.detach().cpu().numpy())
            
            self._global_step += 1
            

        # epoch-end aggregates
        writer.add_scalar("loss/actor_epoch",  _safe_mean(actor_losses), epoch)
        writer.add_scalar("loss/critic_epoch", _safe_mean(critic_losses), epoch)
        writer.add_scalar("policy/logp_pi_epoch", _safe_mean(logp_means), epoch)
        writer.add_scalar("q/q1_mean_epoch", _safe_mean(q1_means), epoch)
        writer.add_scalar("q/q2_mean_epoch", _safe_mean(q2_means), epoch)
        writer.add_scalar("q/qmin_pi_epoch", _safe_mean(qmin_pi_means), epoch)
        if ent_losses:
            writer.add_scalar("loss/entropy_epoch", _safe_mean(ent_losses), epoch)

        if local_writer:
            writer.flush(); writer.close()

        self._epoch_h = None
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
            self._eval_h = self.net.init_hidden(1, self.device)

        feat, h_next = self.net.encode(obs, self._eval_h)

        if is_evaluate:
            mu, _ = self.net._mean_std(feat)
            action = th.tanh(mu)
            q1, q2 = self.net.q(feat, action)
            logp = th.zeros(1, 1, device=self.device)
            v_like = th.min(q1, q2)[0].detach().cpu().numpy()
        else:
            action, logp = self.net.sample_from_features(feat, detach_feat_for_actor=True)
            v_like = 0

        self._eval_h = h_next

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
            'log_alpha': self.log_alpha,  # Save the tensor itself
            'alpha_opt_state_dict': self.alpha_opt.state_dict(),
            'cfg': self.cfg,
            'epoch_h': self._epoch_h,  # Save the hidden state if required
            'global_step': self._global_step,
        }
        th.save(checkpoint, path)


    def load(self, path: str, device: str):
        """Load the model and optimizer states."""
        checkpoint = th.load(path, map_location=device, weights_only=False)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        self.critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])
        
        self.log_alpha = checkpoint['log_alpha'].to(device)  # Directly assign the tensor
        self.alpha_opt.load_state_dict(checkpoint['alpha_opt_state_dict'])
        
        self.cfg = checkpoint['cfg']
        self._epoch_h = checkpoint['epoch_h']
        self._global_step = checkpoint['global_step']
        self.device = device
        self.net.to(device)

