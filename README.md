# Dual WiFi Access Point RL Controller (by Claude)

A deep reinforcement learning framework for intelligent control of dual WiFi access points using advanced RL algorithms including SAC with belief states, RNN for sequential processing, and priority experience replay.

## 📋 Overview

This repository implements a sophisticated RL-based controller for managing dual WiFi access points (APs). The system learns optimal policies for:
- **Channel allocation** between two APs
- **Power control** for interference management
- **Rate adaptation** based on network conditions
- **Resource scheduling** for multiple clients

The framework supports multiple state-of-the-art RL algorithms with a focus on partial observability handling through belief states and temporal modeling using RNNs.

## 🚀 Key Features

### Advanced RL Algorithms
- **SAC with RNN and Belief States** (v12): Latest streamlined implementation
- **SAC with RNN and Belief States** (v11): Previous version with unified feature encoder
- **Distributional RL**: Quantile regression for improved value estimation
- **Priority Experience Replay**: Multiple replay buffer strategies for sample efficiency
- **Meta-Learning**: Support for few-shot adaptation to new environments

### Network Architecture
- **RNN-based Sequential Processing**: GRU networks for temporal dependencies
- **Belief State Modeling**: Probabilistic inference for partial observability
- **Streamlined Architecture** (v12): Direct obs+belief input to actor/critic
- **Multi-Critic Architecture**: Reduces overestimation bias
- **On-the-fly Feature Computation**: No separate feature extraction step

### Performance Optimizations
- **Feature Caching**: Reduces redundant computations
- **Gradient Scheduling**: Different update frequencies for components
- **Memory Management**: Efficient batch processing and state management
- **Distributed Training**: Support for multi-GPU training

## 📁 Repository Structure

```
dual_wifi_AP/
├── agent.py                      # Main agent implementation
├── train_rl.py                   # Training script
├── test.py                       # Testing utilities
├── compute_normalization.py      # Data preprocessing
├── tap.py                        # Trace analysis and processing
├── log_viewer.py                 # Visualization tools
│
├── net_util/                     # Core RL algorithms and utilities
│   ├── model/                    # Neural network architectures
│   │   ├── sac_rnn_belief_seq_dist_v12.py  # Latest streamlined model
│   │   └── sac_rnn_belief_seq_dist_v11.py  # Previous model version
│   ├── algorithms/               # RL policy implementations
│   │   ├── sac_rnn_belief_seq_dist_v12.py  # Latest algorithm
│   │   └── sac_rnn_belief_seq_dist_v11.py  # Previous version
│   ├── replay/                   # Replay buffer implementations
│   ├── base.py                  # Base classes and interfaces
│   └── net_config/              # Configuration files
│       └── 12_10_v6/            # Latest experiment configs
│
├── config/                       # Legacy configuration files
├── util/                         # Utility functions
│   ├── ipc.py                   # Inter-process communication
│   ├── control_cmd.py           # Command interface
│   └── trace_collec.py          # Trace data collection
│
├── exp_trace/                    # Experiment traces
├── logs/                         # Training logs and checkpoints
├── plots/                        # Generated plots and visualizations
└── tools/                        # Additional tools and scripts
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.8.0+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/dual_wifi_AP.git
cd dual_wifi_AP
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Setup environment**
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 🏃 Quick Start

### 1. Training a New Agent

```bash
python train_rl.py \
    --config net_util/net_config/12_10_v6/STA1_STA2.json \
    --output_dir logs/experiment_001
```

### 2. Running an Agent

```bash
python agent.py \
    --config config/network/sac_rnn_belief.py \
    --model_path net_cp/latest.pt \
    --server_ip 127.0.0.1 \
    --server_port 11112
```

### 3. Visualizing Results

```bash
python log_viewer.py --log_dir logs/experiment_001
```

## 📊 Configuration

The system uses JSON-based configuration for flexible experiment setup:

```json
{
    "policy_cfg": {
        "policy_name": "SACRNNBeliefSeqDistV12",
        "batch_size": 256,
        "learning_starts": 2000,
        "gamma": 0.99,
        "tau": 0.005,
        "lr": 0.0003,
        "obs_dim": 6,
        "act_dim": 2,
        "belief_dim": 1,
        "bins": 26,
        "n_critics": 5,
        "burn_in": 50,
        "belief_update_frequency": 5,
        "network_module_path": "net_util.model.sac_rnn_belief_seq_dist_v12"
    },
    "rollout_cfg": {
        "buffer_name": "RNNPriReplayEqualEpFixed",
        "capacity": 10000,
        "episode_length": 600
    },
    "agent_cfg": {
        "server_ip": "127.0.0.1",
        "server_port": 11112,
        "period": 0.005,
        "duration": 20
    }
}
```

## 🧠 Algorithms

### SAC with RNN and Belief States (v12 - Latest)

The most recent implementation (v12) features a streamlined architecture:

1. **Direct Input Processing**
   - Observations and belief states fed directly to actor/critic
   - No separate feature extraction step
   - Cleaner, more efficient architecture

2. **Belief State Modeling**
   - Probabilistic inference for partial observability
   - KL regularization with annealing
   - Interference prediction

3. **Distributional Value Learning**
   - Quantile regression for value distributions
   - Configurable number of critics (default: 5)
   - Target dropout for regularization

4. **Priority Experience Replay**
   - Mixed prioritization (loss + reward)
   - Reduces Q-value collapse
   - Improved convergence

### SAC with RNN and Belief States (v11)

Previous implementation with:

1. **Unified Feature Encoder**
   - Shared GRU encoder with task-specific projections
   - Feature caching for improved efficiency
   - Modular architecture

### Previous Versions

- **v11**: Unified feature encoder with caching
- **v10**: First unified encoder implementation
- **v9**: Modular network architecture
- **v8-v2**: Iterative improvements
- **v1**: Initial SAC+RNN+Belief implementation

## 📈 Performance Metrics

The system tracks comprehensive metrics:

### Training Metrics
- **Losses**: Actor, critic, belief, entropy
- **Value Estimates**: Q-value distributions, TD errors
- **Policy Statistics**: Action distributions, entropy
- **Parameter Norms**: Network parameter tracking

### System Metrics
- **Throughput**: Aggregate and per-client
- **Fairness**: Jain's fairness index
- **Delay**: Packet latency statistics
- **Interference**: Cross-AP interference levels

## 🔧 Customization

### Adding New Algorithms

1. Create policy class in `net_util/`
```python
from net_util.base import PolicyBase

@register_policy
class MyPolicy(PolicyBase):
    def __init__(self, cmd_cls, cfg):
        # Implementation here
        pass
```

2. Register configuration
```python
from net_util import register_policy_cfg

@register_policy_cfg
@dataclass
class MyPolicyConfig:
    param1: int = 64
    param2: float = 0.001
```

### Custom Replay Buffers

Implement replay buffer in `net_util/rnn_replay_custom.py`:
```python
class CustomReplayBuffer:
    def __init__(self, capacity, **kwargs):
        # Implementation
        pass

    def add(self, episode):
        # Add episode logic
        pass

    def sample(self, batch_size):
        # Sampling logic
        pass
```

## 📊 Experiments

### Example Configurations

1. **High Interference Scenario**
```json
{
    "state_cfg": {
        "outage_rate": {"rule": true},
        "interference": {"rule": true}
    },
    "reward_cfg": {
        "acc_bitrate": {"alpha": 2e-07},
        "outage_rate": {"zeta": -4.0}
    }
}
```

2. **Fast Convergence**
```json
{
    "policy_cfg": {
        "lr": 0.001,
        "batch_size": 256,
        "encoder_update_frequency": 1
    }
}
```

### Baselines

The framework includes various baselines:
- Random policy
- Fixed allocation
- Round-robin scheduling
- Proportional fairness

## 🔍 Debugging

### Common Issues

1. **Q-value Collapse**
   - Check replay buffer alpha parameter
   - Monitor priority distribution
   - Use mixed prioritization

2. **Belief Instability**
   - Adjust KL annealing schedule
   - Check interference labels
   - Monitor belief loss

3. **Memory Issues**
   - Reduce batch size
   - Use gradient checkpointing
   - Clear feature cache regularly

### Visualization Tools

- **TensorBoard**: Real-time training monitoring
- **log_viewer.py**: Custom log analysis
- **trace_analysis.py**: Network trace visualization

## 📚 References

1. **Soft Actor-Critic**: Haarnoja et al., 2018
2. **Quantile Regression DQN**: Dabney et al., 2018
3. **Recurrent Deep Q Networks**: Hausknecht & Stone, 2015
4. **Variational Belief State Modeling**: Igl et al., 2019
5. **Prioritized Experience Replay**: Schaul et al., 2015


---

## 📝 Changelog

### v12.0 (Latest)
- Streamlined architecture with direct obs+belief input to actor/critic
- Removed separate feature extraction step for improved efficiency
- Configurable number of critics and quantile bins
- Fixed various bugs and improved stability
- Updated configuration structure with new parameters

### v11.0
- Refactored architecture with utility functions extracted
- Unified feature encoder for improved efficiency
- Enhanced debugging with modular logging
- Comprehensive documentation

### v10.0
- Introduced unified feature encoder
- Improved memory management
- Added feature caching mechanism

### v9.0
- Modular network architecture
- Support for multiple critic networks
- Enhanced belief state modeling

### v1.0-v8.0
- Initial SAC implementation
- RNN integration
- Belief state addition
- Priority replay implementation
- Performance optimizations