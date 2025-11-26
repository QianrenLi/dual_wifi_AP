import math
import torch as th
from torch import Tensor
from typing import Iterable, Callable

def debug_param_change(fn: Callable):
    """
    Decorator: print how much parameters change (global L2 and max |Δw|)
    across a single optimizer step-like function.

    Assumes the first argument is `params_iterable`.
    """
    def wrapper(self, params_iterable: Iterable[th.nn.Parameter], *args, **kwargs):
        # Materialize params because we need them twice (before & after),
        # and clip_grad_norm_ also accepts a list.
        params = list(params_iterable)

        # Snapshot params BEFORE the step
        with th.no_grad():
            before = [p.detach().clone() for p in params]

        # Call the original function, but pass the materialized list
        result = fn(params, *args, **kwargs)

        # Compute stats AFTER the step
        with th.no_grad():
            total_delta_sq = 0.0
            max_delta = 0.0
            for p, b in zip(params, before):
                delta = (p - b).view(-1)
                d_norm = delta.norm().item()
                total_delta_sq += d_norm * d_norm
                max_delta = max(max_delta, delta.abs().max().item())

            total_delta = math.sqrt(total_delta_sq)

        print(
            f"[DEBUG] {fn.__name__}: "
            f"global ||Δw||_2 = {total_delta:.3e}, "
            f"max |Δw_i| = {max_delta:.3e}"
        )

        return result

    return wrapper