"""
Attentive Experience Replay Buffer for RNN-based RL.

This module implements an attention-based experience replay buffer that samples
episodes based on their similarity to the most recent episode. Uses cosine
similarity between the last N timesteps of observations to bias sampling.
"""

from typing import List, Optional, Iterable, Dict
from net_util import register_buffer
import torch as th
import numpy as np
import random
import math


# =============================================================================
# Episode Source Types
# =============================================================================

class EpisodeSource:
    """Source type for episodes - determines eligibility as query episode."""

    CURRENT = "current"          # Online/current experience (used as query)
    BACKGROUND = "background"    # Offline/background dataset (candidates only)
    OFFLINE = "offline"          # Other offline sources (candidates only)


# Import Episode class and helper functions from rnn_fifo
# We reuse the same Episode structure for compatibility
from .rnn_fifo import (
    Episode,
    _as_1d_float,
    _tracify,
    summarize,
    merge,
)


# =============================================================================
# CORE FUNCTIONALITY - Attentive Replay Specific
# =============================================================================

def cosine_similarity_episodes(
    ep1_obs: np.ndarray,
    ep2_obs: np.ndarray,
    n_timesteps: int = 10
) -> float:
    """
    Compute cosine similarity between last N timesteps of two episodes.

    Args:
        ep1_obs: Observations from episode 1, shape (T, obs_dim)
        ep2_obs: Observations from episode 2, shape (T, obs_dim)
        n_timesteps: Number of timesteps from end to use for comparison

    Returns:
        Cosine similarity in range [-1, 1], where 1 means identical
    """
    # Extract last n_timesteps (or fewer if episode is short)
    t1 = min(n_timesteps, ep1_obs.shape[0])
    t2 = min(n_timesteps, ep2_obs.shape[0])

    if t1 == 0 or t2 == 0:
        return 0.0

    # Get last t timesteps and flatten
    obs1 = ep1_obs[-t1:].flatten().astype(np.float32)
    obs2 = ep2_obs[-t2:].flatten().astype(np.float32)

    # Handle edge case of zero vectors
    norm1 = np.linalg.norm(obs1)
    norm2 = np.linalg.norm(obs2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    # Compute cosine similarity
    similarity = np.dot(obs1, obs2) / (norm1 * norm2)
    return float(similarity)


@register_buffer
class AttentiveReplayBuffer:
    """
    Attentive Experience Replay Buffer for RNN-based RL.

    This buffer samples episodes based on their similarity to the most recent
    episode, using cosine similarity between the last N timesteps of observations.

    Sampling strategy:
        1. Sample lambda_val * batch_size candidate episodes uniformly
        2. Compute similarity to the query episode (most recent)
        3. Select the batch_size most similar episodes

    API-compatible with RNNPriReplayFiFo for drop-in replacement.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(
        self,
        device: str = "cuda",
        capacity: int = 10000,
        gamma: float = 0.99,
        alpha: float = 0.7,  # API compatibility only
        beta0: float = 0.4,  # API compatibility only
        rebalance_interval: int = 100,  # API compatibility only
        lambda_val: float = 4.0,  # CORE: sampling multiplier for attention
        n_query_timesteps: int = 10,  # CORE: timesteps to use for similarity
        writer=None,
        episode_length: int = 300,
    ):
        # CORE: Attentive replay parameters
        self.device = th.device(device)
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.lambda_val = float(lambda_val)
        self.n_query_timesteps = int(n_query_timesteps)
        self.episode_length = episode_length

        # CORE: Storage and query tracking
        self.heap: List[Episode] = []
        self._next_insert: int = 0
        self._query_episode: Optional[Episode] = None

        # CORE: Source-aware tracking (distinguishes current vs background episodes)
        self._episode_sources: Dict[int, str] = {}  # Maps episode index -> source
        self.current_ep_idx: Optional[int] = None  # Index of the current episode (used as query)

        # API compatibility: Unused parameters (kept for drop-in replacement)
        self.alpha = float(alpha)
        self.beta = float(beta0)
        self.rebalance_interval = int(rebalance_interval)

        # API compatibility: Statistics tracking (not used for sampling)
        self.writer = writer
        self.reward_summary = (0, 0.0, 0.0)
        self.gamma_summary = (0, 0.0, 0.0)
        self.avg_return = 0.0
        self.run_return = 0.0
        self.data_num = 0
        self.sigma = 0.0

    # -------------------------------------------------------------------------
    # CORE: Attentive Replay Internal Methods
    # -------------------------------------------------------------------------
    def _push(self, ep: Episode, source: str = EpisodeSource.BACKGROUND):
        """
        FIFO insert with source-aware query episode tracking.

        Args:
            ep: Episode to insert
            source: Source type ("current", "background", or "offline")
        """
        # Track evicted episode for cleanup
        evicted_idx = None
        if len(self.heap) >= self.capacity and self.capacity > 0:
            evicted_idx = self._next_insert

        if len(self.heap) < self.capacity:
            ep._heap_idx = len(self.heap)
            self.heap.append(ep)
        else:
            # Overwrite the oldest episode
            self.heap[self._next_insert] = ep
            ep._heap_idx = self._next_insert

        # CORE: Track episode source
        self._episode_sources[ep._heap_idx] = source

        # CORE: Update current episode tracker ONLY if source is "current"
        # New current episode becomes the query (replaces previous)
        if source == EpisodeSource.CURRENT:
            self.current_ep_idx = ep._heap_idx

        # CORE: Cleanup evicted episode from tracking
        if evicted_idx is not None:
            # Remove from episode sources
            if evicted_idx in self._episode_sources:
                del self._episode_sources[evicted_idx]
            # If the current episode was evicted, clear the tracker
            if evicted_idx == self.current_ep_idx:
                self.current_ep_idx = None

        # Advance circular pointer
        if self.capacity > 0:
            self._next_insert = (self._next_insert + 1) % self.capacity

    def _compute_similarities(
        self,
        query_ep: Episode,
        candidate_eps: List[Episode]
    ) -> np.ndarray:
        """
        CORE: Compute cosine similarities between query episode and candidates.

        Args:
            query_ep: The query episode to compare against
            candidate_eps: List of candidate episodes

        Returns:
            Array of similarity scores, shape (len(candidate_eps),)
        """
        if not candidate_eps:
            return np.array([], dtype=np.float32)

        query_obs = query_ep.obs_np
        similarities = np.zeros(len(candidate_eps), dtype=np.float32)

        for i, cand_ep in enumerate(candidate_eps):
            # Skip if candidate is the query episode itself
            if cand_ep is query_ep:
                similarities[i] = -float('inf')
            else:
                similarities[i] = cosine_similarity_episodes(
                    query_obs,
                    cand_ep.obs_np,
                    self.n_query_timesteps
                )

        return similarities

    def _choose_episode_ids(self, batch_size: int):
        """
        CORE: Sample episodes using source-aware attention-based selection.

        Two-phase sampling:
        1. Sample lambda_val * batch_size candidate episodes uniformly
        2. Select batch_size most similar to the query episode (most recent current episode)

        Query episode is the most recent "current" episode.
        Background episodes are never used as query (only as candidates).

        Optimization: When n_candidates <= batch_size, skip similarity computation
        since all candidates will be selected anyway.

        Returns:
            ep_ids: list[int] - selected episode IDs
            probs: None (IS weights are always uniform for attentive replay)
        """
        N = len(self.heap)

        # Edge case: buffer too small for attention-based sampling
        if N <= batch_size:
            # Fall back to uniform sampling of all available episodes
            idxs = np.arange(N, dtype=np.int64)
            chosen = np.random.choice(idxs, size=batch_size, replace=True)
            return chosen.tolist(), None

        # CORE: Use current episode as query if available
        # If no current episode exists, fall back to uniform sampling
        if self.current_ep_idx is not None:
            # Use the tracked current episode as query
            self._query_episode = self.heap[self.current_ep_idx]
        else:
            # No current episode - fall back to uniform sampling
            idxs = np.arange(N, dtype=np.int64)
            chosen = np.random.choice(idxs, size=batch_size, replace=True)
            return chosen.tolist(), None

        # Phase 1: Sample candidates uniformly
        n_candidates = min(int(self.lambda_val * batch_size), N)
        candidate_idxs = np.random.choice(N, size=n_candidates, replace=False)

        # OPTIMIZATION: Skip similarity computation when all candidates will be selected
        # This happens when lambda_val = 1 (n_candidates == batch_size)
        # or when buffer is small (n_candidates < batch_size)
        if n_candidates <= batch_size:
            # Sample with replacement to get exact batch_size
            chosen = np.random.choice(candidate_idxs, size=batch_size, replace=True)
            return chosen.tolist(), None

        # Phase 2: Compute similarities and select top-k (only when n_candidates > batch_size)
        candidate_eps = [self.heap[i] for i in candidate_idxs]
        similarities = self._compute_similarities(
            self._query_episode, candidate_eps
        )

        # Select top-k most similar
        top_k_indices = np.argsort(similarities)[-batch_size:]
        chosen_ids = [candidate_idxs[i] for i in top_k_indices]

        return chosen_ids, None

    # -------------------------------------------------------------------------
    # CORE: Main Sampling Interface
    # -------------------------------------------------------------------------
    def _gather_batch(
        self,
        ep_ids: List[int],
        T: int,
        device: Optional[th.device | str] = None,
    ):
        """
        Assemble batch from episode arrays using np.stack.

        Returns tensors on `device`.
        """
        if not ep_ids:
            return None, None, None, None, None, None

        dev = th.device(device) if device is not None else self.device

        eps = [self.heap[eid] for eid in ep_ids]
        B = len(eps)

        obs_TB = np.stack([ep.obs_np[:T] for ep in eps], axis=1)
        act_TB = np.stack([ep.actions_np[:T] for ep in eps], axis=1)
        rew_TB = np.stack([ep.rewards_np[:T] for ep in eps], axis=1)[..., None]
        nxt_TB = np.stack([ep.next_obs_np[:T] for ep in eps], axis=1)
        done_TB = np.stack([ep.dones_np[:T] for ep in eps], axis=1)[..., None]
        interfs = np.array([ep.interference for ep in eps], dtype=np.float32)
        rets = np.stack([ep.rets_np[:T] for ep in eps], axis=1)[..., None]

        obs_TB_t = th.from_numpy(obs_TB).to(dev, non_blocking=True)
        act_TB_t = th.from_numpy(act_TB).to(dev, non_blocking=True)
        rew_TB_t = th.from_numpy(rew_TB).to(dev, non_blocking=True)
        nxt_TB_t = th.from_numpy(nxt_TB).to(dev, non_blocking=True)
        done_TB_t = th.from_numpy(done_TB).to(dev, non_blocking=True)
        interf_B_t = th.from_numpy(interfs).to(dev, non_blocking=True).unsqueeze(-1)
        rets_TB_t = th.from_numpy(rets).to(dev, non_blocking=True)

        return obs_TB_t, act_TB_t, rew_TB_t, nxt_TB_t, done_TB_t, interf_B_t, rets_TB_t

    def get_sequences(
        self,
        batch_size: int,
        device: Optional[th.device | str] = None,
        *args,
        **kwargs,
    ):
        """
        CORE: Yield a single batch of sequences using attention-based sampling.

        Returns:
            (obs_TBD, act_TBA, rew_TB1, nxt_TBD, done_TB1, info)
        """
        if not self.heap:
            return

        dev = th.device(device) if device is not None else self.device

        ep_ids, probs_all = self._choose_episode_ids(batch_size)
        if not ep_ids:
            return

        idxs = np.asarray(ep_ids, dtype=np.int64)
        probs_ep = probs_all[idxs] if probs_all is not None else np.ones(len(ep_ids)) / len(self.heap)

        obs_TB, act_TB, rew_TB, nxt_TB, done_TB, interf_B, rets_TB_t = self._gather_batch(
            ep_ids, self.episode_length, device=dev
        )

        is_w_B1 = self._calc_is_weights(probs_ep, device=dev)
        prob_B1 = th.from_numpy(probs_ep.astype(np.float32)).to(
            dev, non_blocking=True
        ).unsqueeze(-1)

        info = {
            "ep_ids": ep_ids,
            "interference": interf_B,
            "is_weights": is_w_B1,
            "probs": prob_B1,
            "returns": rets_TB_t,
        }
        yield (obs_TB, act_TB, rew_TB, nxt_TB, done_TB, info)

    # -------------------------------------------------------------------------
    # API COMPATIBILITY: Factory Methods (for RNNPriReplayFiFo compatibility)
    # -------------------------------------------------------------------------
    @staticmethod
    def build_from_traces(
        traces,
        device="cuda",
        reward_agg="sum",
        capacity=10000,
        init_loss: Optional[float] = None,
        episode_length=600,
        lambda_val=4.0,
        n_query_timesteps=10,
        **kwargs
    ):
        """Build buffer from traces (factory method)."""
        buf = AttentiveReplayBuffer(
            device=device,
            capacity=capacity,
            gamma=kwargs.get("gamma", 0.99),
            alpha=kwargs.get("alpha", 0.7),
            beta0=kwargs.get("beta0", 0.4),
            rebalance_interval=kwargs.get("rebalance_interval", 100),
            lambda_val=lambda_val,
            n_query_timesteps=n_query_timesteps,
            writer=kwargs.get("writer", None),
            episode_length=episode_length
        )
        interfs = kwargs.get("interference_vals", [0] * len(traces))
        for (states, actions, rewards, network_output), interf in zip(traces, interfs):
            buf.add_episode(
                states, actions, rewards, network_output,
                reward_agg, init_loss, interf
            )
        return buf

    # -------------------------------------------------------------------------
    # API COMPATIBILITY: Add Episodes (with statistics tracking)
    # -------------------------------------------------------------------------
    def extend(
        self,
        traces,
        reward_agg="sum",
        init_loss: Optional[float] = None,
        source: str = EpisodeSource.BACKGROUND,  # NEW: source parameter
        **kwargs
    ):
        """Extend buffer with multiple traces."""
        interfs = kwargs.get("interference_vals", [0] * len(traces))
        for (states, actions, rewards, network_output), interf in zip(traces, interfs):
            self.add_episode(
                states, actions, rewards, network_output,
                reward_agg, init_loss, interf, source
            )

    def add_episode(
        self,
        states,
        actions,
        rewards,
        network_output,
        reward_agg="sum",
        init_loss: Optional[float] = None,
        interference=0,
        source: str = EpisodeSource.BACKGROUND  # NEW: source parameter
    ):
        """Add a single episode to the buffer."""
        obs_np, act_np, rew_np, next_obs_np, done_np = _tracify(
            states, actions, rewards, network_output, reward_agg
        )

        # Decompose the trace into multiple episodes of the specified length
        num_episodes = len(obs_np) // self.episode_length
        for i in range(num_episodes):
            start = i * self.episode_length
            end = (i + 1) * self.episode_length

            ep = Episode(
                obs_np[start:end],
                act_np[start:end],
                rew_np[start:end],
                next_obs_np[start:end],
                done_np[start:end],
                init_loss=10000.0 if init_loss is None else float(init_loss),
                gamma=self.gamma,
                interference=interference
            )

            # API compatibility: Update running stats (not used for sampling)
            self.reward_summary = merge(self.reward_summary, ep.reward_summary)
            self.gamma_summary = merge(self.gamma_summary, ep.gamma_summary)

            for r in ep.rewards_np.tolist():
                self.run_return = float(r) + self.gamma * self.run_return
                self.data_num += 1
                if self.writer is not None:
                    self.writer.add_scalar("data/return", self.run_return, self.data_num)

            # FIFO insert with source-aware query tracking
            self._push(ep, source)

    def add_preprocessed_episodes(
        self,
        episodes: List[Episode],
        source: str = EpisodeSource.BACKGROUND  # NEW: source parameter
    ):
        """Fast insertion method for pre-processed Episode objects."""
        for ep in episodes:
            self._push(ep, source)

    # -------------------------------------------------------------------------
    # API COMPATIBILITY: No-op Methods (not used for attentive replay)
    # -------------------------------------------------------------------------
    def update_episode_losses(
        self,
        ep_ids: List[int],
        losses: Iterable[float]
    ):
        """
        API compatibility: No-op for attentive replay.
        (Sampling based on similarity, not loss-based priority)
        """
        pass

    def _calc_is_weights(
        self,
        probs_ep: np.ndarray,
        beta: Optional[float] = None,
        device: Optional[th.device | str] = None,
    ) -> th.Tensor:
        """
        API compatibility: Returns uniform IS weights.
        (Attentive replay doesn't need bias correction)
        """
        if probs_ep is None or probs_ep.size == 0:
            return th.zeros((0, 1), dtype=th.float32, device=device or self.device)

        wi = np.ones_like(probs_ep, dtype=np.float32)
        w_t = th.from_numpy(wi)
        dev = th.device(device) if device is not None else self.device
        w_t = w_t.to(dev, non_blocking=True)
        return w_t.unsqueeze(-1)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def length(self):
        """Return number of episodes in buffer."""
        return len(self.heap)
