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

    def _ppo_update(self):
        clip = self.cfg.clip_range
        ent_coef, vf_coef = self.cfg.ent_coef, self.cfg.vf_coef
        max_grad_norm = self.cfg.max_grad_norm
        buf = self.buf
        
        # Advantage normalization (per update)
        adv = buf.advantages.view(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buf.advantages = adv.view(buf.size, buf.n_envs)

        for epoch in range(self.cfg.n_epochs):
            approx_kls = []
            for obs, act, old_logp, adv, ret, old_v in buf.get_minibatches(self.cfg.batch_size):
                values, logp, entropy = self.net.evaluate_actions(obs, act)
                ratio = th.exp(logp - old_logp)

                # Policy loss (clipped surrogate)
                pg1 = adv * ratio
                pg2 = adv * th.clamp(ratio, 1.0 - clip, 1.0 + clip)
                policy_loss = -th.min(pg1, pg2).mean()

                # Value loss (no vf clipping here for brevity)
                value_loss = F.mse_loss(ret, values)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                # KL approx for early stop
                with th.no_grad():
                    log_ratio = logp - old_logp
                    approx_kl = ((th.exp(log_ratio) - 1) - log_ratio).mean().item()
                    approx_kls.append(approx_kl)

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_grad_norm)
                self.opt.step()

            if self.cfg.target_kl is not None:
                if np.mean(approx_kls) > 1.5 * self.cfg.target_kl:
                    break

    def train_per_epoch(self, log_path: str = "net_util/logs/train.log"):
        with open(log_path, "a") as f:
            for upd in range(self.cfg.n_updates):
                self._ppo_update()
                # Simple logging
                with th.no_grad():
                    ev = 1.0 - th.var((self.buf.returns - self.buf.values).view(-1)) / (
                        th.var(self.buf.returns.view(-1)) + 1e-8
                    )
                    avg_ret = self.buf.returns.view(-1).mean().item()
                    msg = (f"[{upd+1:04d}/{self.cfg.n_updates:04d}] "
                        f"avg_return={avg_ret:8.3f}  "
                        f"exp_var={ev.item():6.3f}  "
                        f"log_std={self.net.log_std.data.mean():6.3f}\n")
                f.write(msg)
                f.flush()   # ensures it's written to disk

    @th.no_grad()
    def tf_act(self, obs_vec):
        obs = th.tensor(obs_vec, device=self.device).float()
        action, log_prob, value = self.net(obs)
        return {
            "action": action.cpu().numpy(),
            "log_prob": log_prob.cpu().numpy(),
            "value": value.cpu().numpy()
        }

