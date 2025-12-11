# Net Util Refactoring Guide

## Overview
The `net_util` package has been refactored to achieve clean separation between RL algorithms and replay buffer implementations. This improves maintainability, flexibility, and reduces coupling between components.

## New Structure

```
net_util/
├── __init__.py                 # Main registry and exports
├── base.py                     # PolicyBase (unchanged)
├── rollout.py                  # Rollout utilities (unchanged)
├── advantage_estimator.py     # Advantage estimators (unchanged)
├── state_transfom.py          # State transformations (unchanged)
├── legacy.py                   # Backward compatibility layer
│
├── replay/                    # Replay buffer implementations
│   ├── __init__.py           # Buffer registry & exports
│   ├── base.py               # BaseReplayBuffer interface
│   ├── simple.py             # ReplayBuffer (from replay.py)
│   ├── rnn_priority.py       # RNNPriReplayBuffer2 (priority replay)
│   └── specialized/          # Specialized replay buffers
│
├── algorithms/               # RL algorithm implementations
│   ├── __init__.py          # Algorithm registry & exports
│   ├── sac_belief_seq.py    # SAC with RNN and belief states
│   └── ppo.py               # PPO algorithm
│
└── model/                    # Neural network architectures (unchanged)
    └── ...
```

## Key Changes

### 1. Replay Buffer Registry
All replay buffers are now registered in a central registry:

```python
from net_util import get_buffer_class, BUFFER_REGISTRY

# Get buffer class by name
buffer_cls = get_buffer_class('RNNPriReplayBuffer2')

# Create buffer instance
buffer = buffer_cls(device='cuda', recent_capacity=10, top_capacity=300)
```

### 2. Dynamic Buffer Selection
Algorithms now create buffers dynamically from configuration:

```python
class SACAlgorithm(PolicyBase):
    def __init__(self, cfg):
        # Buffer name from config
        buffer_name = cfg.rollout_cfg.get('buffer_name', 'RNNPriReplayBuffer2')

        # Create buffer from registry
        buffer_cls = get_buffer_class(buffer_name)
        self.buffer = buffer_cls(**cfg.rollout_cfg)
```

### 3. Standardized Buffer Interface
All buffers implement `BaseReplayBuffer` with required methods:
- `add_episode(states, actions, rewards, network_output)`
- `sample(batch_size, **kwargs)`
- `update_priorities(indices, priorities)` (if supported)
- `size()`, `ready(min_size)`

### 4. Backward Compatibility
Legacy imports continue to work through compatibility layer:
```python
# Old imports still work
from net_util.rnn_replay_with_pri_2 import RNNPriReplayBuffer2
from net_util.replay import ReplayBuffer
```

## Migration Guide

### For Buffer Users
No changes needed if using the legacy imports. For new code:

```python
# Old way
from net_util.rnn_replay_with_pri_2 import RNNPriReplayBuffer2
buffer = RNNPriReplayBuffer2(device='cuda')

# New way
from net_util import get_buffer_class
buffer_cls = get_buffer_class('RNNPriReplayBuffer2')
buffer = buffer_cls(device='cuda')
```

### For Algorithm Developers
Use the registry for buffer selection:

```python
class MyAlgorithm(PolicyBase):
    def __init__(self, cmd_cls, cfg, **kwargs):
        super().__init__(cmd_cls, **kwargs)

        # Get buffer from config
        buffer_name = cfg.rollout_cfg['buffer_name']
        self.buffer = self._create_buffer(buffer_name, cfg.rollout_cfg)

    def _create_buffer(self, name, cfg):
        from net_util import get_buffer_class
        buffer_cls = get_buffer_class(name)
        return buffer_cls(**cfg)
```

### Configuration Format
JSON configs specify buffer name in `rollout_cfg`:

```json
{
  "rollout_cfg": {
    "buffer_name": "RNNPriReplayBuffer2",
    "recent_capacity": 20,
    "top_capacity": 500,
    "alpha": 0.3,
    "beta0": 0.4
  }
}
```

## Benefits

1. **Clean Separation**: Algorithms and buffers are in separate modules
2. **Flexibility**: Easy to swap buffers without modifying algorithms
3. **Maintainability**: Related code grouped together
4. **Extensibility**: Simple to add new algorithms or buffers
5. **Backward Compatibility**: Existing configs continue to work

## Adding New Components

### New Replay Buffer
1. Create file in `replay/` (e.g., `replay/my_buffer.py`)
2. Inherit from `BaseReplayBuffer`
3. Implement required methods
4. Add `@register_buffer` decorator
5. Import in `replay/__init__.py`

```python
from net_util import register_buffer
from .base import BaseReplayBuffer

@register_buffer
class MyBuffer(BaseReplayBuffer):
    def __init__(self, capacity: int, **kwargs):
        super().__init__(capacity, **kwargs)
        # Implementation
```

### New Algorithm
1. Create file in `algorithms/` (e.g., `algorithms/my_algorithm.py`)
2. Use registry for buffer selection
3. Add `@register_policy` and `@register_policy_cfg` decorators
4. Import in `algorithms/__init__.py`

## Status

- ✅ Completed: Core infrastructure
- ✅ Completed: Replay buffer migration (ReplayBuffer, RNNPriReplayBuffer2)
- ✅ Completed: Algorithm migration (PPO, SAC variants)
- ✅ Completed: Backward compatibility layer
- 🔄 In Progress: Migrating remaining algorithms
- 📋 TODO: Migrating specialized replay buffers