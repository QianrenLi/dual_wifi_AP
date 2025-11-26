from typing import List
import torch as th
from util.trace_collec import _apply_rule

class ValueDistribution:
    def __init__(
        self,
        reward_cfg: dict,
        scale: float = 1.1,
        bins: int = 51,
        gamma: float = 0.99,
    ):
        self.bins = bins
        self.bounds = self.get_bound(reward_cfg, gamma, scale=scale)
        self.value_bins = self.get_bin_value()   # python list
        self._value_bins_t: th.Tensor | None = None  # cached tensor version

    # --------- internal helpers --------- #
    def _value_bins_tensor(self, device=None, dtype=None) -> th.Tensor:
        """Return value_bins as a Torch tensor on the requested device/dtype."""
        if self._value_bins_t is None:
            self._value_bins_t = th.tensor(self.value_bins, dtype=th.float32)
        v = self._value_bins_t
        if device is not None:
            v = v.to(device)
        if dtype is not None and v.dtype != dtype:
            v = v.to(dtype)
        return v

    def _project_tensor(self, Tz: th.Tensor, dist: th.Tensor) -> th.Tensor:
        """
        C51-style projection of transformed support Tz back onto self.value_bins.

        Tz   : (..., K)
        dist : (..., K)  (probabilities over original support)
        returns: (..., K)
        """
        assert Tz.shape == dist.shape
        device = dist.device
        dtype = dist.dtype

        v = self._value_bins_tensor(device=device, dtype=dtype)  # [K]
        v0 = v[0]
        vN = v[-1]
        K = self.bins
        delta = (vN - v0) / (K - 1)

        # Clamp transformed support into bounds
        Tz = Tz.clamp(min=v0, max=vN)  # (..., K)

        # Flatten leading dims into one batch dim
        orig_shape = Tz.shape            # (*S, K)
        B = Tz.numel() // K
        Tz_flat   = Tz.view(B, K)        # [B, K]
        dist_flat = dist.view(B, K)      # [B, K]

        # Fractional bin positions
        b = (Tz_flat - v0) / delta       # [B, K]
        l = b.floor().long()
        u = b.ceil().long()

        l = l.clamp(0, K - 1)
        u = u.clamp(0, K - 1)

        target_flat = th.zeros_like(dist_flat)               # [B, K]
        offset = th.arange(B, device=device).unsqueeze(1) * K  # [B, 1]

        # Lower index contribution
        target_flat.view(-1).index_add_(
            0,
            (l + offset).view(-1),
            (dist_flat * (u.float() - b)).view(-1),
        )
        # Upper index contribution
        target_flat.view(-1).index_add_(
            0,
            (u + offset).view(-1),
            (dist_flat * (b - l.float())).view(-1),
        )

        return target_flat.view(orig_shape)

    # --------- NEW: simplified soft target --------- #
    def soft_target_distribution(
        self,
        dist: th.Tensor,        # (..., K)
        reward: th.Tensor,      # (..., 1)
        d_TB1: th.Tensor,      # (..., 1)
        gamma: float,           # scalar
        alpha_logp: th.Tensor,  # (..., 1)
    ) -> th.Tensor:
        """
        Compute soft Bellman target distribution:

            z' = r + gamma * (z - alpha_logp)

        and C51-project it back onto self.value_bins.

        Shapes:
            dist      : (..., K)   (probabilities over self.value_bins)
            reward    : (..., 1)
            alpha_logp: (..., 1)
            gamma     : scalar
        """
        assert dist.shape[-1] == self.bins
        assert reward.shape[-1] == 1
        assert alpha_logp.shape[-1] == 1
        assert dist.shape[:-1] == reward.shape[:-1] == alpha_logp.shape[:-1] == d_TB1.shape[:-1] 

        device = dist.device
        dtype = dist.dtype
        K = self.bins

        # Base support z_i
        v = self._value_bins_tensor(device=device, dtype=dtype)  # [K]
        # Expand to (..., K)
        v_expanded = v.view(*([1] * (dist.dim() - 1)), K)        # (..., K)

        # z_soft = z - alpha_logp
        z_soft = v_expanded - alpha_logp                         # (..., K)

        # Tz = r + gamma * z_soft
        Tz = reward + (1 - d_TB1) * gamma * z_soft                             # (..., K)

        # Project back using original distribution dist
        return self._project_tensor(Tz, dist)
    
    
    # --------- public API --------- #
    def mean_value(self, value_distribution):
        """
        If value_distribution is:
        - torch.Tensor with shape (..., bins) -> returns tensor with shape (..., 1)
        - list/np array of length bins       -> returns float
        """
        if isinstance(value_distribution, th.Tensor):
            # value_distribution: [..., K], K == self.bins
            v = self._value_bins_tensor(
                device=value_distribution.device,
                dtype=value_distribution.dtype,
            )  # [K]
            # broadcast multiply and sum over last dim
            return (value_distribution * v).sum(dim=-1, keepdim=True)
        else:
            # Fallback to Python list / numpy
            mean_val = 0.0
            for p, val in zip(value_distribution, self.value_bins):
                mean_val += p * val
            return float(mean_val)

    def get_bin_value(self) -> list:
        bin_values = []
        step = (self.bounds[1] - self.bounds[0]) / (self.bins - 1)
        for i in range(self.bins):
            bin_values.append(self.bounds[0] + i * step)
        return bin_values

    def get_bound(self, reward_cfg, gamma, scale) -> tuple:
        """
        Calculate the value function bounds based on reward configuration.
        """
        MAX_BITRATE = 3e7  # 30 Mbps
        MAX_OUTAGE = 1.0   # 100% outage
        max_bound = 0.0
        min_bound = 0.0

        # Handle bitrate
        entry = reward_cfg.get("bitrate", {})
        if entry:  # avoid `is not {}` bug
            rule = entry.get("rule", True)
            args = entry.get("args")
            bitrate_bound = _apply_rule(MAX_BITRATE, rule, args)
            max_bound = bitrate_bound / (1 - gamma)

        # Handle outage
        entry = reward_cfg.get("outage_rate", {})
        if entry:
            rule = entry.get("rule", True)
            args = entry.get("args")
            outage_bound = _apply_rule(MAX_OUTAGE, rule, args)
            min_bound = outage_bound / (1 - gamma)

        span = max_bound - min_bound
        mean = (min_bound + max_bound) / 2.0
        return (mean - span * scale / 2.0, mean + span * scale / 2.0)

if __name__ == "__main__":
    REWARD_DESCRIPTOR = {
        "bitrate": {
            "rule": "bitrate_delta",
            "from": "bitrate",                 # bare key -> flow["bitrate"]
            "args": {"alpha": 1e-6 * 0.05 * 0.02, "beta": -1e-6 / 2 * 0.05 * 0.02, "offset": 0.05},
            "pos": "flow",
        },
        "outage_rate": {
            "rule": "scale_outage",
            "from": "outage_rate",             # bare key -> flow["outage_rate"]
            "args": {"zeta": -1000 * 0.05 * 0.02 },
            "pos": "flow",
        }
    }
    
    vd = ValueDistribution(REWARD_DESCRIPTOR, bins=51, gamma=0.99)
    
    print(vd.bounds)