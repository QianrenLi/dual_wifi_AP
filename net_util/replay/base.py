"""Base interface for all replay buffers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class BaseReplayBuffer(ABC):
    """Standard interface for all replay buffers."""

    def __init__(self, capacity: int, device: str = "cuda", **kwargs):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of episodes to store
            device: Device to store tensors on
            **kwargs: Additional buffer-specific parameters
        """
        self.capacity = capacity
        self.device = device

    @abstractmethod
    def add_episode(self, states, actions, rewards, network_output, interference: int = 0):
        """
        Add a complete episode to the buffer.

        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            network_output: List of network outputs (may include done flags)
            interference: Interference level (optional)
        """
        pass

    @abstractmethod
    def sample(self, batch_size: int, **kwargs):
        """
        Sample a batch from the buffer.

        Args:
            batch_size: Number of episodes to sample
            **kwargs: Additional sampling parameters

        Returns:
            Sampled batch of episodes/transitions
        """
        pass

    @abstractmethod
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled indices (if supported).

        Args:
            indices: Indices to update
            priorities: New priority values
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Return current buffer size."""
        pass

    @abstractmethod
    def ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size() >= min_size

    @abstractmethod
    def extend(self, traces: List[Tuple], device: str, **kwargs):
        """
        Extend buffer with multiple traces.

        Args:
            traces: List of (states, actions, rewards, network_output) tuples
            device: Device to move data to
            **kwargs: Additional parameters
        """
        pass

    def build_from_traces(cls, traces: List[Tuple], device: str, **kwargs):
        """
        Alternative constructor to build buffer from traces.

        Args:
            traces: List of (states, actions, rewards, network_output) tuples
            device: Device to move data to
            **kwargs: Additional parameters

        Returns:
            Initialized buffer with traces added
        """
        # Extract buffer capacity from kwargs or use default
        capacity = kwargs.get('buffer_max', 10000)
        buffer = cls(capacity=capacity, device=device, **kwargs)

        # Add all traces
        if traces:
            buffer.extend(traces, device, **kwargs)

        return buffer