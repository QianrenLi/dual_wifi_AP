from dataclasses import dataclass
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import importlib

from net_util.rollout import RolloutBuffer
from net_util.base import PolicyBase
from . import register_policy, register_policy_cfg

@register_policy_cfg
@dataclass
class PPO_Config:
    # total_steps: int = 200_000
    n_updates: int = 1000
    n_envs: int = 16
    obs_dim: int = 6
    act_dim: int = 2
    rollout_len: int = 128
    n_epochs: int = 8
    batch_size: int = 1024
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    device: str = "cpu"
    seed: int = 0
    target_kl: float = 0.03  # None to disable
    
    network_module_path = "net_util.model.ac"

def _safe_mean(xs):
    return float(np.mean(xs)) if xs else float("nan")

def symlog(x: th.Tensor, eps=1e-12) -> th.Tensor:
    return th.sign(x) * th.log(th.abs(x) + 1.0 + eps)

@register_policy
class PPO(PolicyBase):
    def __init__(self, cmd_cls, cfg: PPO_Config, rollout_buffer: RolloutBuffer = None, device = None):
        super().__init__(cmd_cls)
        th.manual_seed(cfg.seed); np.random.seed(cfg.seed)
        self.cfg = cfg
        self.device = th.device(cfg.device) if device is None else th.device(device)
        
        ## load network module
        net_mod = importlib.import_module(cfg.network_module_path)
        Network = getattr(net_mod, "Network")
        
        self.net = Network(cfg.obs_dim, cfg.act_dim).to(self.device)
        self.opt = th.optim.Adam(self.net.parameters(), lr=cfg.lr)
        
        self.buf = rollout_buffer

    def _ppo_update(self, f):
        clip = self.cfg.clip_range
        ent_coef, vf_coef = self.cfg.ent_coef, self.cfg.vf_coef
        max_grad_norm = self.cfg.max_grad_norm
        buf = self.buf

        for epoch in range(self.cfg.n_epochs):
            # Per-epoch meters
            policy_loss_meter, value_loss_meter, entropy_meter = [], [], []
            total_loss_meter, kl_meter, clipfrac_meter = [], [], []

            approx_kls = []
            for obs, act, old_logp, adv, ret, old_v in buf.get_minibatches(self.cfg.batch_size):
                values, logp, entropy = self.net.evaluate_actions(obs, act)
                ratio = th.exp(logp - old_logp)

                # PPO losses
                pg1 = adv * ratio
                pg2 = adv * th.clamp(ratio, 1.0 - clip, 1.0 + clip)
                policy_loss = -th.min(pg1, pg2).mean()
                value_loss = F.mse_loss(symlog(ret), symlog(values))
                entropy_loss = -entropy.mean()
                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                # Metrics (KL & clipfrac)
                with th.no_grad():
                    log_ratio = logp - old_logp
                    approx_kl = ((th.exp(log_ratio) - 1) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)
                    clipped = ((ratio > (1.0 + clip)) | (ratio < (1.0 - clip))).float().mean().item()

                # Step
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
                self.opt.step()

                # Collect per-minibatch stats
                policy_loss_meter.append(policy_loss.item())
                value_loss_meter.append(value_loss.item())
                entropy_meter.append(entropy.mean().item())
                total_loss_meter.append(loss.item())
                kl_meter.append(approx_kl)
                clipfrac_meter.append(clipped)

            # Early stop on KL (per-epoch)
            if self.cfg.target_kl is not None and np.mean(approx_kls) > 1.5 * self.cfg.target_kl:
                early_stop = True
            else:
                early_stop = False

            # Per-epoch log extras (exp var / avg return / log_std)
            with th.no_grad():
                ev = 1.0 - th.var((buf.returns - buf.values).view(-1)) / (th.var(buf.returns.view(-1)) + 1e-8)
                avg_ret = buf.returns.view(-1).mean().item()
                try:
                    lr = next(iter(self.opt.param_groups))["lr"]
                except Exception:
                    lr = float("nan")

            # ---- Write ONE line per epoch ----
            msg = (
                f"[epoch {epoch+1:03d}/{self.cfg.n_epochs:03d}] "
                f"avg_return={avg_ret:8.3f}  "
                f"exp_var={ev.item():6.3f}  "
                f"log_std={self.net.log_std.data.mean():6.3f}  "
                f"loss={_safe_mean(total_loss_meter):8.5f}  "
                f"pol_loss={_safe_mean(policy_loss_meter):8.5f}  "
                f"val_loss={_safe_mean(value_loss_meter):8.5f}  "
                f"entropy={_safe_mean(entropy_meter):7.5f}  "
                f"kl={_safe_mean(kl_meter):7.5f}  "
                f"clipfrac={_safe_mean(clipfrac_meter):6.3f}  "
                f"lr={lr:.3e}"
            )
            if early_stop:
                msg += "  [early_stop:KL]"
            f.write(msg + "\n")
            f.flush()

            if early_stop:
                break

    def train_per_epoch(self, log_path: str = "net_util/logs/train.log"):
        with open(log_path, "w") as f:
            self._ppo_update(f)

    @th.no_grad()
    def tf_act(self, obs_vec, is_evaluate = False):
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

    def save(self, path:str):
        th.save({
            "model": self.net.state_dict(),
            "optimizer": self.opt.state_dict(),
        }, path)

        
    def load(self, path: str, device:str):
        ckpt = th.load(path, map_location=device)

        self.net.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])

        self.net.to(device)