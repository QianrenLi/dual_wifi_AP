import torch as th
import numpy as np
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

        eq_mask = (l == u)  # [B, K]
        
        target_flat.view(-1).index_add_(
            0,
            (l + offset)[eq_mask].view(-1),
            dist_flat[eq_mask].view(-1)
        )

        # 3. Case l != u: Standard C51 linear interpolation
        neq_mask = ~eq_mask
        
        # Lower index contribution (only where l != u)
        target_flat.view(-1).index_add_(
            0,
            (l + offset)[neq_mask].view(-1),
            (dist_flat * (u.float() - b))[neq_mask].view(-1),
        )
        
        # Upper index contribution (only where l != u)
        target_flat.view(-1).index_add_(
            0,
            (u + offset)[neq_mask].view(-1),
            (dist_flat * (b - l.float()))[neq_mask].view(-1),
        )

        return target_flat.view(orig_shape)

    # --------- NEW: simplified soft target --------- #
    def soft_target_distribution(
        self,
        dist: th.Tensor,        # (..., K)
        reward: th.Tensor,      # (..., 1)
        d_TB1: th.Tensor,       # (..., 1)
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
    def mean_value(
        self,
        value_distribution,
        q_low: float | None = None,
        q_high: float | None = None,
    ):
        """
        If value_distribution is:
        - torch.Tensor with shape (..., bins) -> returns tensor with shape (..., 1)
        - list/np array of length bins       -> returns float

        If q_low, q_high are given (e.g. 0.05, 0.95), compute the mean value
        restricted to the quantile interval [q_low, q_high] (approximate by
        slicing the CDF along the bin axis).
        """
        # --------- torch path --------- #
        if isinstance(value_distribution, th.Tensor):
            probs = value_distribution
            # v: [K]
            v = self._value_bins_tensor(
                device=probs.device,
                dtype=probs.dtype,
            )

            # No quantile restriction -> standard mean
            if q_low is None or q_high is None or (q_low <= 0.0 and q_high >= 1.0):
                return (probs * v).sum(dim=-1, keepdim=True)

            # Clamp / sanity
            assert q_low >= 0.0 and q_high <= 1.0
            assert q_low < q_high, ValueError(f"q_high ({q_high}) must be > q_low ({q_low}).")

            # CDF along last dim
            cdf = probs.cumsum(dim=-1)                           # [..., K]
            prev_cdf = th.cat(
                [th.zeros_like(cdf[..., :1]), cdf[..., :-1]],
                dim=-1
            )                                                    # [..., K]

            start = prev_cdf
            end   = cdf

            # Length of intersection of [start,end] with [q_low, q_high]
            left  = end.clamp(max=q_high)                      # min(end, q_high)
            right = start.clamp(min=q_low)                     # max(start, q_low)
            seg   = (left - right).clamp(min=0.0)                # [..., K]

            denom = max(q_high - q_low, 1e-8)
            weights = seg / denom                                # [..., K], sum≈1

            # Conditional mean on that quantile slice
            return (weights * v).sum(dim=-1, keepdim=True)

        # --------- Python list / numpy path --------- #
        else:
            probs = np.asarray(value_distribution, dtype=np.float64)
            v = np.asarray(self.value_bins, dtype=np.float64)

            # No quantile restriction -> standard mean
            if q_low is None or q_high is None or (q_low <= 0.0 and q_high >= 1.0):
                return float(np.sum(probs * v))

            q_low = float(max(0.0, min(1.0, q_low)))
            q_high = float(max(0.0, min(1.0, q_high)))
            if q_high <= q_low:
                raise ValueError(f"q_high ({q_high}) must be > q_low ({q_low}).")

            # Normalize to be safe
            probs = probs / (probs.sum() + 1e-12)

            cdf = np.cumsum(probs)                # [K]
            prev_cdf = np.concatenate(([0.0], cdf[:-1]))

            start = prev_cdf
            end   = cdf

            left  = np.minimum(end, q_high)
            right = np.maximum(start, q_low)
            seg   = np.clip(left - right, 0.0, None)

            denom = max(q_high - q_low, 1e-8)
            weights = seg / denom

            return float(np.sum(weights * v))
        
    def mean_minus_k_sigma(self, value_distribution, k: float):
        if isinstance(value_distribution, th.Tensor):
            probs = value_distribution
            v = self._value_bins_tensor(
                device=probs.device,
                dtype=probs.dtype,
            )
            v2 = v * v

            mean = (probs * v).sum(dim=-1, keepdim=True)
            second_moment = (probs * v2).sum(dim=-1, keepdim=True)
            var = (second_moment - mean.pow(2)).clamp_min(1e-6)
            std = var.sqrt()

            k_t = th.as_tensor(k, device=probs.device, dtype=probs.dtype)
            return mean - k_t * std

        else:
            probs = np.asarray(value_distribution, dtype=np.float64)
            v = np.asarray(self.value_bins, dtype=np.float64)
            v2 = v * v

            mean = float(np.sum(probs * v))
            second_moment = float(np.sum(probs * v2))
            var = max(second_moment - mean * mean, 0.0)
            std = float(np.sqrt(var))

            return mean - k * std
        

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