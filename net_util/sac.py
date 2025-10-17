from dataclasses import dataclass
from typing import Optional, Tuple
import copy
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import importlib

from net_util.base import PolicyBase
from net_util.replay import ReplayBuffer  # provided elsewhere
from . import register_policy, register_policy_cfg
from torch.utils.tensorboard import SummaryWriter


def _safe_mean(xs):
    return float(np.mean(xs)) if xs else float("nan")

def symlog(x: th.Tensor, eps=1e-12) -> th.Tensor:
    return th.sign(x) * th.log(th.abs(x) + 1.0 + eps)

def hard_update(dst: nn.Module, src: nn.Module):
    for p, q in zip(dst.parameters(), src.parameters()):
        p.data.copy_(q.data)

@th.no_grad()
def soft_update(dst: nn.Module, src: nn.Module, tau: float):
    for p, q in zip(dst.parameters(), src.parameters()):
        p.data.mul_(1.0 - tau).add_(q.data, alpha=tau)


@register_policy_cfg
@dataclass
class SAC_Config:
    # schedule
    n_updates: int = 1000
    gradient_steps: int = 1
    batch_size: int = 256

    # dims
    n_envs: int = 1
    obs_dim: int = 6
    act_dim: int = 2

    # replay
    buffer_size: int = 200_000
    learning_starts: int = 2_000

    # alg
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4

    # entropy temperature (alpha)
    ent_coef: str | float = "auto"            # "auto" or float
    ent_coef_init: float = 1.0
    target_entropy: str | float = "auto"      # "auto" -> -act_dim
    target_update_interval: int = 1

    # misc
    device: str = "cpu"
    seed: int = 0

    # networks
    network_module_path: str = "net_util.model.sac_model"  # must expose Network(obs_dim, act_dim)

@register_policy
class SAC(PolicyBase):
    """
    Soft Actor-Critic with twin Q critics and target networks.

    Expectations for `cfg.network_module_path`:
      - Exposes class `Network(obs_dim, act_dim)` with:
          .actor: stochastic actor exposing:
              - __call__(obs) -> (action, logp, value_like_or_aux) for eval mode (deterministic OK)
              - act(obs)      -> (action, logp, value_like_or_aux) for training (reparameterized)
          .critic or .critic1: Q(obs, act) -> (B, 1)
          .critic_target or .critic1_target (optional)

      If only a single critic is provided, this class will:
          - Create .critic1 from provided .critic
          - Deep-copy to build .critic2
          - Create matching target networks for both
    """

    def __init__(self, cmd_cls, cfg: SAC_Config, rollout_buffer: ReplayBuffer | None = None, device: str | None = None, state_transform_dict = None):
        super().__init__(cmd_cls, state_transform_dict = state_transform_dict)
        th.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.cfg = cfg
        self.device = th.device(cfg.device) if device is None else th.device(device)

        # Load network module
        net_mod = importlib.import_module(cfg.network_module_path)
        Network = getattr(net_mod, "Network")
        self.net: nn.Module = Network(cfg.obs_dim, cfg.act_dim, scale_log_offset = self.cmd_cls.sum_log_scales()).to(self.device)

        # --- actor ---
        self.actor: nn.Module = self.net.actor

        # --- critics (robust extraction/synthesis) ---
        # preferred names
        critic1 = getattr(self.net, "critic1", getattr(self.net, "critic", None))
        if critic1 is None:
            raise AttributeError("Network must define `.critic` or `.critic1`.")
        self.critic1: nn.Module = critic1

        critic2 = getattr(self.net, "critic2", None)
        if critic2 is None:
            # Synthesize critic2 by deep-copy of critic1
            self.critic2 = copy.deepcopy(self.critic1)
            # Attach back to net for checkpoint coherence
            setattr(self.net, "critic2", self.critic2)
        else:
            self.critic2: nn.Module = critic2

        # Targets: prefer provided, else create as deep-copies
        c1_t = getattr(self.net, "critic1_target", getattr(self.net, "critic_target", None))
        if c1_t is None:
            self.critic1_target = copy.deepcopy(self.critic1)
            setattr(self.net, "critic1_target", self.critic1_target)
        else:
            self.critic1_target: nn.Module = c1_t

        c2_t = getattr(self.net, "critic2_target", None)
        if c2_t is None:
            self.critic2_target = copy.deepcopy(self.critic2)
            setattr(self.net, "critic2_target", self.critic2_target)
        else:
            self.critic2_target: nn.Module = c2_t

        self.net.to(self.device)
        # Hard-sync targets
        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)

        # Optimizers
        self.actor_opt = th.optim.Adam(list(self.actor.parameters()) \
                    + list(self.net.log_std_net.parameters()) \
                    + [self.net.log_std_bias], lr=cfg.lr)
        
        # Include BOTH critics' parameters
        self.critic_opt = th.optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=cfg.lr)

        # Temperature α
        self.target_entropy = (- float(cfg.act_dim)) if cfg.target_entropy == "auto" else float(cfg.target_entropy)
        self.log_alpha: Optional[th.Tensor] = None
        self.alpha_opt: Optional[th.optim.Adam] = None
        if cfg.ent_coef == "auto":
            self.log_alpha = th.log(th.ones(1, device=self.device) * cfg.ent_coef_init).requires_grad_(True)
            self.alpha_opt = th.optim.Adam([self.log_alpha], lr=cfg.lr)
        else:
            self.alpha_tensor = th.tensor(float(cfg.ent_coef), device=self.device)

        # Replay buffer handle (external)
        self.buf = rollout_buffer

        # Meters
        self._steps_seen = 0
        self._n_updates = 0
        self._global_step = 0

    # ---- Data API (optional helpers) ----
    def add_transition(self, obs, act, rew, next_obs, done):
        self.buf.add(obs, act, rew, next_obs, done)
        self._steps_seen += 1

    # ---- Acting ----
    @th.no_grad()
    def tf_act(self, obs_vec, is_evaluate: bool = False):
        """
        Returns dict with action/log_prob and a lightweight 'value' = min(Q1,Q2)(obs, action)
        """
        obs = th.tensor(obs_vec, device=self.device).float()
        if is_evaluate:
            action, log_prob, v_like = self.net(obs)
            action = th.tanh(action)
            qa = self.critic1(th.cat([obs, action], dim=-1))
            qb = self.critic2(th.cat([obs, action], dim=-1))
            v_like = th.min(qa, qb).detach().cpu().numpy()
        else:
            action, log_prob, v_like = self.net.act(obs)

        return {
            "action": action.detach().cpu().numpy(),
            "log_prob": log_prob.detach().cpu().numpy(),
            "value": v_like,
        }

    # ---- Training loop ----
    def train_per_epoch(self, epoch, log_dir: str = "runs/sac", writer: Optional[SummaryWriter] = None):
        """
        Train for one epoch over `self.cfg.gradient_steps` *minibatches* and log metrics.
        """
        local_writer = False
        if writer is None:
            writer = SummaryWriter(log_dir=log_dir)
            local_writer = True

        # Buffer status
        buf_ptr = getattr(self.buf, "ptr", 0)
        buf_size = getattr(self.buf, "size", 0)
        writer.add_scalar("buffer/ptr", buf_ptr, epoch)
        writer.add_scalar("buffer/size", buf_size, epoch)

        if buf_ptr < max(self.cfg.learning_starts, self.cfg.batch_size):
            writer.add_text("status", f"waiting_for_data size={buf_ptr}", epoch)
            if local_writer:
                writer.flush(); writer.close()
            return 0.0  # keep API compatible

        # Stats
        ent_losses, ent_vals = [], []
        actor_losses, critic_losses = [], []
        q1_vals, q2_vals, qmin_vals = [], [], []
        log_pi_vals = []

        # Iterate over gradient steps (pulling minibatches from your buffer API)
        steps_done = 0
        for obs, act, rew, nxt, done in self.buf.get_minibatches(self.cfg.batch_size):
            if steps_done >= self.cfg.gradient_steps:
                return False
            
            steps_done += 1

            # ----- Temperature α update -----
            if self.alpha_opt is not None and self.log_alpha is not None:
                with th.no_grad():
                    a_pi_for_alpha, logp_pi_for_alpha, _ = self.net.act(obs)
                alpha = self.log_alpha.detach().exp()
                ent_loss = -(self.log_alpha * (logp_pi_for_alpha + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad()
                ent_loss.backward()
                self.alpha_opt.step()
                ent_losses.append(ent_loss.item()); ent_vals.append(alpha.item())
                writer.add_scalar("sac/alpha_step", alpha.item(), self._global_step)
            else:
                alpha = self.alpha_tensor
                ent_losses.append(float("nan")); ent_vals.append(alpha.item())
                writer.add_scalar("sac/alpha_step", alpha.item(), self._global_step)

            # ----- Critic targets (Double Q, min over target critics) -----
            with th.no_grad():
                a_next, logp_next, _ = self.net.act(nxt)
                q1_t = self.critic1_target(th.cat([nxt, a_next], dim=-1))
                q2_t = self.critic2_target(th.cat([nxt, a_next], dim=-1))
                q_next_min = th.min(q1_t, q2_t)
                backup = rew + (1.0 - done) * self.cfg.gamma * (q_next_min - alpha * logp_next.reshape(-1, 1))

            # ----- Critic update (sum of MSEs) -----
            q1_pred = self.critic1(th.cat([obs, act], dim=-1))
            q2_pred = self.critic2(th.cat([obs, act], dim=-1))
            c_loss = F.mse_loss(symlog(q1_pred), symlog(backup)) + F.mse_loss(symlog(q2_pred), symlog(backup))
            self.critic_opt.zero_grad()
            c_loss.backward()
            self.critic_opt.step()

            # ----- Actor update (reparameterization, minimize KL ~ maximize Q - alpha*logpi) -----
            a_pi, logp_pi, _ = self.net.act(obs)
            q1_pi = self.critic1(th.cat([obs, a_pi], dim=-1))
            q2_pi = self.critic2(th.cat([obs, a_pi], dim=-1))
            q_pi_min = symlog(th.min(q1_pi, q2_pi))
            a_loss = (alpha * logp_pi - q_pi_min).mean()
            self.actor_opt.zero_grad()
            a_loss.backward()
            self.actor_opt.step()

            # ----- Target updates -----
            if (self._n_updates % self.cfg.target_update_interval) == 0:
                soft_update(self.critic1_target, self.critic1, self.cfg.tau)
                soft_update(self.critic2_target, self.critic2, self.cfg.tau)

            # meters
            self._n_updates += 1
            self._global_step += 1
            actor_losses.append(a_loss.item())
            critic_losses.append(c_loss.item())
            q1_vals.append(q1_pred.mean().item())
            q2_vals.append(q2_pred.mean().item())
            qmin_vals.append(q_pi_min.mean().item())
            log_pi_vals.append(logp_pi.detach().mean().item())

        # LR / epoch aggregates
        try:
            lr = next(iter(self.actor_opt.param_groups))["lr"]
        except Exception:
            lr = float("nan")

        writer.add_scalar("loss/actor_epoch", _safe_mean(actor_losses), epoch)
        writer.add_scalar("loss/critic_epoch", _safe_mean(critic_losses), epoch)
        writer.add_scalar("loss/entropy_epoch", _safe_mean(ent_losses), epoch)
        writer.add_scalar("q/q1_mean_epoch", _safe_mean(q1_vals), epoch)
        writer.add_scalar("q/q2_mean_epoch", _safe_mean(q2_vals), epoch)
        writer.add_scalar("q/qmin_pi_epoch", _safe_mean(qmin_vals), epoch)
        writer.add_scalar("policy/logp_pi_epoch", _safe_mean(log_pi_vals), epoch)
        writer.add_scalar("opt/lr_epoch", lr, epoch)

        if local_writer:
            writer.flush()
            writer.close()

        return True  # keep signature stable

    # ---- Checkpointing ----
    def save(self, path: str):
        state = {
            "model": self.net.state_dict(),
            "actor_optimizer": self.actor_opt.state_dict(),
            "critic_optimizer": self.critic_opt.state_dict(),
            "log_alpha": (self.log_alpha.detach().cpu() if self.log_alpha is not None else None),
            "alpha_opt": (self.alpha_opt.state_dict() if self.alpha_opt is not None else None),
            "global_step": self._global_step,
            "n_updates": self._n_updates,
        }
        th.save(state, path)

    def load(self, path: str, device: str):
        ckpt = th.load(path, map_location=device)

        self.net.load_state_dict(ckpt["model"])
        self.actor_opt.load_state_dict(ckpt["actor_optimizer"])
        self.critic_opt.load_state_dict(ckpt["critic_optimizer"])

        # Restore alpha (robust to None/old checkpoints)
        if ckpt.get("log_alpha", None) is not None:
            if self.log_alpha is None:
                # If current config uses fixed alpha but checkpoint had auto,
                # create the tensor so we can load & then ignore going forward.
                self.log_alpha = th.tensor(0.0, requires_grad=True, device=device)
            self.log_alpha.data.copy_(ckpt["log_alpha"].to(device))
        if self.alpha_opt is not None and ckpt.get("alpha_opt", None) is not None:
            self.alpha_opt.load_state_dict(ckpt["alpha_opt"])

        self._global_step = ckpt.get("global_step", 0)
        self._n_updates = ckpt.get("n_updates", 0)

        self.net.to(device)
