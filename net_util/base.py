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
        state_transform_dict: Optional[dict] = None,
        reward_cfg: Optional[dict] = None,
    ):
        self.cmd_cls = cmd_cls
        self.action_dim: int = cmd_cls.__dim__()
        if seed is not None:
            random.seed(seed)
        self.net = None
        self.opt = None
        self.reward_cfg = reward_cfg
        
        self._state_tf: Optional[_StateTransform] = None
        if state_transform_dict:
            try:
                self._state_tf = _StateTransform.from_obj(state_transform_dict)
            except Exception as e:
                # Fall back silently (or log if you prefer)
                print(f"[PolicyBase] Failed to load state transform: {e}")
                self._state_tf = None

    def tf_act(self, obs_vec: List[float], is_evaluate: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

    # --- agent API ---
    def act(self, obs: List, is_evaluate: bool = False) -> List[float]:
        obs = self._pre_act(obs)
        res = self.tf_act(obs, is_evaluate)
        safe_res = {str(k): (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()}
        return safe_res, list_to_cmd(self.cmd_cls, safe_res['action'])

    def _pre_act(self, obs: List) -> List[float]:
        """
        1) Flatten nested obs into a single vector.
        2) If a state transform is configured, normalize it.
        """
        obs = self._state_tf.apply_to_list(obs)
        return obs

    def train_per_epoch(self, epoch, writer=None, is_batch_rl = False):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str, device: str):
        raise NotImplementedError
