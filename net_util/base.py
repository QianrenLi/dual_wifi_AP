import random
from typing import Any, Dict, List, Optional, Type

from util.control_cmd import ControlCmd, list_to_cmd, _json_default  # noqa: E402
from util.trace_collec import flatten_leaves

class PolicyBase:
    """
    Base policy that understands the action structure via ControlCmd class.
    Provides canonical mappings:
      - cmd_to_vec(ControlCmd) -> List[float]
      - vec_to_cmd(List[float], version) -> ControlCmd
    """

    def __init__(self, cmd_cls: Type[ControlCmd], seed: Optional[int] = None):
        self.cmd_cls = cmd_cls
        self.action_dim: int = cmd_cls.__dim__()
        if seed is not None:
            random.seed(seed)

    def tf_act(self, obs_vec: List[float]) -> Dict[str, Any]:
        """TensorFlow action method to be implemented by subclasses."""
        raise NotImplementedError

    # --- policy API ---
    def act(self, obs: Dict[str, Any]) -> List[float]:
        """Deterministic base action (vector of length action_dim)."""
        res = self.tf_act(self._pre_act(obs))
        safe_res = {str(k): (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()}
        return safe_res, list_to_cmd(self.cmd_cls, safe_res['action'])
        
    
    def _pre_act(self, obs: Dict[str, Any]) -> List[float]:
        """Preprocess obs if needed (default: noop)."""
        return flatten_leaves(obs)