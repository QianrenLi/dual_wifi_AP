from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import importlib

from net_util.base import PolicyBase
from net_util.replay import ReplayBuffer  # <-- imported, not implemented here
from . import register_policy, register_policy_cfg


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
    ent_coef: str | float = "auto"     # "auto", or float
    ent_coef_init: float = 1.0
    target_entropy: str | float = "auto"  # "auto" -> -act_dim
    target_update_interval: int = 1

    # misc
    device: str = "cpu"
    seed: int = 0

    # networks
    network_module_path: str = "net_util.model.sac_model"


@register_policy
class SAC(PolicyBase):
    """
    Minimal SAC:
    - Follows your PPO file shape (config+registry, single-line logs, tf_act API).
    - Imports ReplayBuffer from net_util.replay (no local implementation).
    - Requires `SACActor` and `SACCritic` in cfg.network_module_path.
    """

    def __init__(self, cmd_cls, cfg: SAC_Config, rollout_buffer: ReplayBuffer | None = None, device: str | None = None):
        super().__init__(cmd_cls)
        th.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.cfg = cfg
        self.device = th.device(cfg.device) if device is None else th.device(device)

        # Networks from your module
        net_mod = importlib.import_module(cfg.network_module_path)
        Network = getattr(net_mod, "Network")
        
        self.net: nn.Module = Network(cfg.obs_dim, cfg.act_dim).to(self.device)

        self.actor = self.net.actor
        self.critic = self.net.critic
        self.critic_target = self.net.critic_target
        
        hard_update(self.critic_target, self.critic)

        self.actor_opt = th.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = th.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # Alpha
        self.target_entropy = (-float(cfg.act_dim) if cfg.target_entropy == "auto"
                               else float(cfg.target_entropy))
        
        self.log_alpha: Optional[th.Tensor] = None
        self.alpha_opt: Optional[th.optim.Adam] = None
        
        if cfg.ent_coef == "auto":
            self.log_alpha = th.log(th.ones(1, device=self.device) * cfg.ent_coef_init).requires_grad_(True)
            self.alpha_opt = th.optim.Adam([self.log_alpha], lr=cfg.lr)
        else:
            self.alpha_tensor = th.tensor(float(cfg.ent_coef), device=self.device)

        # Replay (imported). If none passed, create one; user will push transitions.
        self.buf = rollout_buffer if rollout_buffer is not None else ReplayBuffer(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            size=cfg.buffer_size,
            device=self.device,
        )
        
        self._steps_seen = 0
        self._n_updates = 0

    # ---- Data API (optional helpers) ----
    def add_transition(self, obs, act, rew, next_obs, done):
        self.buf.add(obs, act, rew, next_obs, done)
        self._steps_seen += 1

    # ---- Acting ----
    @th.no_grad()
    def tf_act(self, obs_vec, is_evaluate: bool = False):
        obs = th.tensor(obs_vec, device=self.device).float()
        if is_evaluate:
            action, log_prob, value = self.net(obs)
        else:
            action, log_prob, value = self.net.act(obs)
        return {
            "action": action.cpu().numpy(),
            "log_prob": log_prob.cpu().numpy(),
            "value": value.cpu().numpy()
        }

    # ---- Training loop (mirrors PPO.train_per_epoch format) ----
    def train_per_epoch(self, log_path: str = "net_util/logs/train.log"):
        with open(log_path, "w") as f:
            for epoch in range(self.cfg.n_updates):
                if getattr(self.buf, "ptr") < max(self.cfg.learning_starts, self.cfg.batch_size):
                    f.write(f"[epoch {epoch+1:03d}/{self.cfg.n_updates:03d}] waiting_for_data size={self.buf.ptr}\n")
                    f.flush()
                    continue

                ent_losses, ent_vals = [], []
                actor_losses, critic_losses = [], []
                q1_vals = []

                for obs, act, rew, nxt, done in self.buf.get_minibatches(self.cfg.batch_size):
                    # alpha
                    if self.alpha_opt is not None and self.log_alpha is not None:
                        with th.no_grad():
                            a_pi, logp_pi, _ = self.net.act(obs)
                        alpha = self.log_alpha.exp().detach()
                        ent_loss = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
                        self.alpha_opt.zero_grad(); ent_loss.backward(); self.alpha_opt.step()
                        ent_losses.append(ent_loss.item()); ent_vals.append(alpha.item())
                    else:
                        alpha = self.alpha_tensor
                        ent_losses.append(float("nan")); ent_vals.append(alpha.item())

                    # critic targets
                    with th.no_grad():
                        a_next, logp_next, _ = self.net.act(nxt)
                        q = self.critic_target(th.cat([nxt, a_next], dim=1))
                        q_next = q - alpha * logp_next.reshape(-1, 1)
                        targ = rew + (1.0 - done) * self.cfg.gamma * q_next

                    # critic update
                    q1_pred = self.critic(th.cat([obs, act], dim=1) )
                    c_loss = F.mse_loss(symlog(q1_pred), symlog(targ))
                    self.critic_opt.zero_grad()
                    c_loss.backward()
                    self.critic_opt.step()

                    # actor update
                    a_pi, logp_pi, _ = self.net.act(obs)
                    q_pi = self.critic_target(th.cat([obs, a_pi], dim=1))
                    a_loss = (alpha * logp_pi - q_pi).mean()
                    self.actor_opt.zero_grad(); a_loss.backward(); self.actor_opt.step()

                    # target update
                    if (self._n_updates % self.cfg.target_update_interval) == 0:
                        soft_update(self.critic_target, self.critic, self.cfg.tau)

                    self._n_updates += 1

                    # meters
                    actor_losses.append(a_loss.item())
                    critic_losses.append(c_loss.item())
                    q1_vals.append(q1_pred.mean().item())

                try:
                    lr = next(iter(self.actor_opt.param_groups))["lr"]
                except Exception:
                    lr = float("nan")

                msg = (
                    f"[epoch {epoch+1:03d}/{self.cfg.n_updates:03d}] "
                    f"buf={self.buf.size:6d}  "
                    f"n_updates={self._n_updates:7d}  "
                    f"alpha={_safe_mean(ent_vals):.4f}  "
                    f"ent_loss={_safe_mean(ent_losses):8.5f}  "
                    f"actor_loss={_safe_mean(actor_losses):8.5f}  "
                    f"critic_loss={_safe_mean(critic_losses):8.5f}  "
                    f"q1={_safe_mean(q1_vals):8.3f}  "
                    f"lr={lr:.3e}"
                )
                f.write(msg + "\n"); f.flush()
                
                ## Early stop if alpha = 0
                if _safe_mean(ent_vals) < 0.0005:
                    break

    def save(self, path:str):
        th.save({
            "model": self.net.state_dict(),
            "actor_optimizer": self.actor_opt.state_dict(),
            "critic_optimizer": self.critic_opt.state_dict(),
        }, path)

        
    def load(self, path: str, device:str):
        ckpt = th.load(path, map_location=device)

        self.net.load_state_dict(ckpt["model"])
        self.actor_opt.load_state_dict(ckpt["actor_optimizer"])
        self.critic_opt.load_state_dict(ckpt["critic_optimizer"])

        self.net.to(device)