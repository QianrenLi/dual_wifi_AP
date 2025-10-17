# policy_base.py (your PolicyBase file; patch below)
import random

from typing import Any, Dict, List, Optional, Type

from util.control_cmd import ControlCmd, list_to_cmd, _json_default  # noqa: E402
from util.trace_collec import flatten_leaves
from net_util.state_transfom import _StateTransform

class PolicyBase:
    """
    Base policy that understands the action structure via ControlCmd class.
    Provides canonical mappings:
      - cmd_to_vec(ControlCmd) -> List[float]
      - vec_to_cmd(List[float], version) -> ControlCmd

    New:
      - optional state transform json; if provided, _pre_act() normalizes state.
    """

    def __init__(
        self,
        cmd_cls: Type[ControlCmd],
        seed: Optional[int] = None,
        state_transform_json: Optional[str] = None
    ):
        self.cmd_cls = cmd_cls
        self.action_dim: int = cmd_cls.__dim__()
        if seed is not None:
            random.seed(seed)
        self.net = None
        self.opt = None

        self._state_tf: Optional[_StateTransform] = None
        if state_transform_json:
            try:
                self._state_tf = _StateTransform.from_json(state_transform_json)
            except Exception as e:
                # Fall back silently (or log if you prefer)
                print(f"[PolicyBase] Failed to load state transform: {e}")
                self._state_tf = None

    def tf_act(self, obs_vec: List[float], is_evaluate: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

    # --- policy API ---
    def act(self, obs: Dict[str, Any], is_evaluate: bool = False) -> List[float]:
        vector = self._pre_act(obs)
        res = self.tf_act(vector, is_evaluate)
        safe_res = {str(k): (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()}
        return safe_res, list_to_cmd(self.cmd_cls, safe_res['action'])

    def _pre_act(self, obs: Dict[str, Any]) -> List[float]:
        """
        1) Flatten nested obs into a single vector.
        2) If a state transform is configured, normalize it.
        """
        vec = flatten_leaves(obs)  # -> List[float]
        print(vec)
        if self._state_tf is not None:
            vec = self._state_tf.apply_to_list(vec)
            print(vec)
        return vec

    def train_per_epoch(self, epoch, writer=None):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str, device: str):
        raise NotImplementedError
