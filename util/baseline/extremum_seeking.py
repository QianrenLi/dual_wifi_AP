from . import register_baseline
from ..trace_collec import trace_filter, flatten_leaves
import numpy as np
import logging
import os

@register_baseline
class ExtremumSeeking:
    def __init__(self, state_cfg, initial_cmd, reward_cfg=None):
        self.state_cfg = state_cfg
        self.reward_cfg = reward_cfg if reward_cfg else {}

        # Initialize control signal 'u'
        self.u = np.array(initial_cmd.copy() if initial_cmd else [0.5, 0.5], dtype=float)
        self.initial_cmd = self.u.copy()

        # SPSA parameters
        self.a = 0.01  # Initial gain coefficient (scaled for reward range [-10, 10])
        self.c = 0.05  # Perturbation coefficient
        self.A = 100   # Stabilization parameter
        self.alpha = 0.000602  # Gain decay exponent
        self.beta = 0.000101   # Perturbation decay exponent

        # Discount factor for cumulative reward
        self.gamma = 0.95  # Discount factor

        # Current estimate of discounted reward
        self.J_discounted = 0.0

        # History buffer for slow-varying second parameter (affects every ~150 iterations)
        self.history_buffer_size = 300  # Buffer for ~2 cycles of slow parameter
        self.slow_param_history = []    # Store (u2_value, reward) pairs
        self.slow_param_window = 150    # Number of iterations for slow parameter effect

        # Bounds for adaptive gains
        self.min_step_size = np.array([0.001] * self.u.size, dtype=float)
        self.max_step_size = np.array([0.05] * self.u.size, dtype=float)
        self.min_delta = np.array([0.01] * self.u.size, dtype=float)
        self.max_delta = np.array([0.1] * self.u.size, dtype=float)

        # Control bounds
        self.u_min = np.array([-1.0] * self.u.size, dtype=float)
        self.u_max = np.array([1.0] * self.u.size, dtype=float)

        # Random number generator
        seed = state_cfg.get("seed", None) if isinstance(state_cfg, dict) else None
        self.rng = np.random.default_rng(seed)

        # Setup logging (can be disabled)
        # self.logging_enabled = state_cfg.get("logging_enabled", True) if isinstance(state_cfg, dict) else True
        self.logging_enabled = False
        if self.logging_enabled:
            self._setup_logging(state_cfg)

        # SPSA state variables
        self.phase = 0  # 0: evaluate at u + c*delta, 1: evaluate at u - c*delta, 2: update
        self.delta_k = None  # Perturbation vector
        self.J_plus = None   # Cost at u + c*delta
        self.J_minus = None  # Cost at u - c*delta
        self.J_prev = None   # Previous cost for improvement tracking
        self.c_k = None      # Perturbation magnitude (preserved between phases)

        # Improvement history for adaptive gain updates
        self.improvement_history = []
        self.window_size = 10

        self.iteration = 0

    def _log(self, level, message):
        """ Safe logging method that checks if logging is enabled """
        if self.logging_enabled and hasattr(self, 'logger'):
            getattr(self.logger, level)(message)

    def _setup_logging(self, state_cfg):
        """ Setup logging for extremum seeking parameters """
        # Create logs directory if it doesn't exist
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(f"ExtremumSeeking_{id(self)}")

        # Configure logging level - change to stop logs
        # OPTIONS: logging.DEBUG (all logs), logging.INFO (most logs),
        #          logging.WARNING (minimal logs), logging.ERROR (errors only)
        log_level = getattr(state_cfg, "log_level", "INFO") if isinstance(state_cfg, dict) else "INFO"
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler for logging
        log_file = os.path.join(log_dir, "extremum_seeking.log")
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Log initial parameters
        self.logger.info("=" * 60)
        self.logger.info("Extremum Seeking SPSA Initialized")
        self.logger.info(f"Initial control signal u: {self.u}")
        self.logger.info(f"SPSA parameters: a={self.a}, c={self.c}, A={self.A}")
        self.logger.info(f"Gain exponents: alpha={self.alpha}, beta={self.beta}")
        self.logger.info(f"Discount factor gamma: {self.gamma}")
        self.logger.info("=" * 60)

    def act(self, current_stats, **kwargs):
        """ Perform the action of mixed-time-scale SPSA extremum seeking algorithm """
        self.iteration += 1

        # Get current reward from system state
        reward = flatten_leaves(trace_filter(current_stats, self.reward_cfg))
        reward = np.asarray(reward, dtype=float)
        immediate_reward = float(np.mean(reward)) if reward.size > 0 else 0.0

        # Update discounted reward estimate
        self.J_discounted = self.gamma * self.J_discounted + (1 - self.gamma) * immediate_reward

        # Update history buffer for slow parameter
        self._update_history_buffer(self.u[1], self.J_discounted)

        # Log reward information
        if self.logging_enabled:
            self.logger.debug(f"Iteration {self.iteration}: immediate_reward={immediate_reward:.4f}, "
                            f"J_discounted={self.J_discounted:.4f}")

        # Handle mixed-time-scale optimization
        if self.u.size >= 2:
            return self._mixed_time_scale_act()
        else:
            # Fall back to standard SPSA for single parameter
            return self._standard_spsa_act()

    def _update_history_buffer(self, u2_value, reward):
        """Update history buffer for slow parameter tracking"""
        self.slow_param_history.append((u2_value, reward))

        # Keep buffer size limited
        if len(self.slow_param_history) > self.history_buffer_size:
            self.slow_param_history.pop(0)

    def _estimate_slow_parameter_gradient(self):
        """Estimate gradient for second parameter using history buffer"""
        if len(self.slow_param_history) < self.slow_param_window:
            return 0.0  # Not enough history

        # Get recent history
        recent_history = self.slow_param_history[-self.slow_param_window:]

        # Group by u2 values to find different perturbation levels
        u2_groups = {}
        for u2_val, reward_val in recent_history:
            # Round u2_val to group similar values
            u2_rounded = round(u2_val, 3)
            if u2_rounded not in u2_groups:
                u2_groups[u2_rounded] = []
            u2_groups[u2_rounded].append(reward_val)

        # If we have at least 2 different u2 values, estimate gradient
        if len(u2_groups) >= 2:
            u2_values = sorted(u2_groups.keys())
            if len(u2_values) >= 2:
                # Get the two most different u2 values
                u2_low, u2_high = u2_values[0], u2_values[-1]
                reward_low = np.mean(u2_groups[u2_low])
                reward_high = np.mean(u2_groups[u2_high])

                # Estimate gradient using finite differences
                if abs(u2_high - u2_low) > 1e-6:
                    gradient_slow = (reward_high - reward_low) / (u2_high - u2_low)
                    return gradient_slow

        return 0.0

    def _mixed_time_scale_act(self):
        """Mixed-time-scale SPSA optimization for two parameters"""
        # Current control signals
        u1_current = self.u[0]  # Fast parameter (affects every iteration)
        u2_current = self.u[1]  # Slow parameter (affects every ~150 iterations)

        # Phase 0: Evaluate perturbation
        if self.phase == 0:
            # Perturb both parameters simultaneously
            delta_k = self.rng.choice([-1.0, 1.0], size=2)
            self.delta_k = delta_k

            # Compute perturbation magnitude and preserve it for phase 1
            self.c_k = self.c / ((self.iteration + 1) ** self.beta)
            self.c_k = np.clip(self.c_k, self.min_delta[0], self.max_delta[0])

            # Store discounted reward at u + c*delta
            self.J_plus = self.J_discounted

            # Apply perturbation to both parameters
            u_apply = self.u.copy()
            u_apply[0] = self._clip_single_dim(u_apply[0] - self.c_k * delta_k[0], 0)
            u_apply[1] = self._clip_single_dim(u_apply[1] - self.c_k * delta_k[1], 1)

            if self.logging_enabled:
                self.logger.debug(f"Phase 0 - u1={u1_current:.4f}, u2={u2_current:.4f}, "
                                f"delta_k={delta_k}, c_k={self.c_k:.6f}, u_apply={u_apply}")

            self.phase = 1
            return u_apply.tolist()

        # Phase 1: Update fast parameter, estimate slow parameter gradient
        elif self.phase == 1:
            # Store discounted reward at u - c*delta
            self.J_minus = self.J_discounted

            if self.J_plus is None or not np.isfinite(self.J_plus):
                self.J_plus = self.J_minus

            # Compute SPSA gain sequence for fast parameter
            a_k_fast = self.a / ((self.iteration/2 + self.A + 1) ** self.alpha)
            a_k_fast = np.clip(a_k_fast, self.min_step_size[0], self.max_step_size[0])

            # Estimate gradient using preserved perturbation magnitude
            gradient_est = (self.J_plus - self.J_minus) / (2.0 * self.c_k)
            g_fast = gradient_est * self.delta_k  # This affects both parameters

            # Estimate additional gradient for slow parameter from history
            g_slow = self._estimate_slow_parameter_gradient()

            # Update both parameters
            u_old = self.u.copy()

            # Update both parameters with SPSA gradient (both affected by fast gradient)
            self.u[0] = self._clip_single_dim(self.u[0] + a_k_fast * g_fast[0], 0)
            self.u[1] = self._clip_single_dim(self.u[1] + a_k_fast * g_fast[1], 1)

            # Add additional update to slow parameter from history-based gradient (larger step size)
            a_k_slow = a_k_fast * 2.0  # Larger step size for slow parameter gradient
            self.u[1] = self._clip_single_dim(self.u[1] + a_k_slow * g_slow, 1)

            # Calculate gradient magnitudes
            gradient_magnitude_fast = np.linalg.norm(g_fast)
            gradient_magnitude_slow = abs(g_slow)

            if self.logging_enabled:
                self.logger.info(f"MIXED_UPDATE - Iteration {self.iteration//2}: "
                               f"J_plus={self.J_plus:.4f}, J_minus={self.J_minus:.4f}")
                self.logger.info(f"FAST_GRADIENT - g_fast={g_fast}, "
                               f"magnitude={gradient_magnitude_fast:.6f}, a_k_fast={a_k_fast:.6f}")
                self.logger.info(f"SLOW_GRADIENT - g_slow={g_slow:.6f}, "
                               f"magnitude={gradient_magnitude_slow:.6f}, a_k_slow={a_k_slow:.6f}, history_size={len(self.slow_param_history)}")
                self.logger.info(f"STEP_UPDATE - u_old={u_old}, u_new={self.u}")

            # Track improvement for adaptive gain adjustment
            J_current = (self.J_plus + self.J_minus) / 2.0
            if self.J_prev is not None and np.isfinite(self.J_prev) and np.isfinite(J_current):
                improved = J_current > self.J_prev
                self._update_gains(improved)
            self.J_prev = J_current

            # Reset for next SPSA iteration
            self.J_plus = None
            self.J_minus = None
            self.phase = 0

            # Apply next perturbation to both parameters
            delta_k = self.rng.choice([-1.0, 1.0], size=2)

            # Update c_k for the next perturbation (will be used in next phase 0)
            self.c_k = self.c / ((self.iteration/2 + 1) ** self.beta)
            self.c_k = np.clip(self.c_k, self.min_delta[0], self.max_delta[0])

            u_apply = self.u.copy()
            u_apply[0] = self._clip_single_dim(u_apply[0] + self.c_k * delta_k[0], 0)
            u_apply[1] = self._clip_single_dim(u_apply[1] + self.c_k * delta_k[1], 1)

            if self.logging_enabled:
                self.logger.debug(f"Next perturbation - delta_k={delta_k}, "
                                f"c_k={self.c_k:.6f}, u_apply={u_apply}")

            return u_apply.tolist()

        # Fallback
        return self.u.tolist()

    def _standard_spsa_act(self):
        """Standard SPSA for single parameter optimization"""
        # Phase 0: Generate perturbation and evaluate at u + c*delta
        if self.phase == 0:
            # Generate random perturbation vector with ±1 entries
            self.delta_k = self._generate_perturbation()

            # Compute perturbation magnitude and preserve for phase 1
            self.c_k = self.c / ((self.iteration + 1) ** self.beta)
            self.c_k = np.clip(self.c_k, self.min_delta[0], self.max_delta[0])

            # Store discounted reward at u + c*delta
            self.J_plus = self.J_discounted

            # Apply u - c*delta perturbation for next evaluation
            u_apply = self._clip_u(self.u - self.c_k * self.delta_k)
            self.phase = 1
            return u_apply.tolist()

        # Phase 1: Store cost at u - c*delta and update
        elif self.phase == 1:
            # Store discounted reward at u - c*delta
            self.J_minus = self.J_discounted

            if self.J_plus is None or not np.isfinite(self.J_plus):
                self.J_plus = self.J_minus

            # Compute SPSA gain sequence
            a_k = self.a / ((self.iteration/2 + self.A + 1) ** self.alpha)
            a_k = np.clip(a_k, self.min_step_size[0], self.max_step_size[0])

            # SPSA gradient estimation using preserved c_k
            gradient_est = (self.J_plus - self.J_minus) / (2.0 * self.c_k)
            g_k = gradient_est * self.delta_k

            # Update control signal using gradient ascent
            u_old = self.u.copy()
            self.u = self._clip_u(self.u + a_k * g_k)

            # Track improvement for adaptive gain adjustment
            J_current = (self.J_plus + self.J_minus) / 2.0
            if self.J_prev is not None and np.isfinite(self.J_prev) and np.isfinite(J_current):
                improved = J_current > self.J_prev
                self._update_gains(improved)
            self.J_prev = J_current

            # Reset for next SPSA iteration
            self.J_plus = None
            self.J_minus = None
            self.phase = 0

            # Generate new perturbation for next iteration
            self.delta_k = self._generate_perturbation()

            # Update c_k for next iteration
            self.c_k = self.c / ((self.iteration/2 + 1) ** self.beta)
            self.c_k = np.clip(self.c_k, self.min_delta[0], self.max_delta[0])

            u_apply = self._clip_u(self.u + self.c_k * self.delta_k)
            return u_apply.tolist()

        # Fallback
        return self.u.tolist()

    def _clip_single_dim(self, value, dim):
        """Clip a single dimension of the control signal"""
        return np.minimum(np.maximum(value, self.u_min[dim]), self.u_max[dim])

    def _generate_perturbation(self):
        """ Generate random perturbation vector with ±1 entries for SPSA """
        return self.rng.choice([-1.0, 1.0], size=self.u.size)

    def _clip_u(self, u):
        """ Clip control signal 'u' to the allowed bounds """
        u = np.asarray(u, dtype=float)
        return np.minimum(np.maximum(u, self.u_min), self.u_max)

    def _update_gains(self, improved):
        """ Update the SPSA gain coefficients based on recent improvements """
        self.improvement_history.append(bool(improved))
        if len(self.improvement_history) > self.window_size:
            self.improvement_history.pop(0)

        if len(self.improvement_history) >= 5:
            rate = sum(self.improvement_history[-5:]) / 5.0

            a_old = self.a
            c_old = self.c

            if rate > 0.7:
                # Increase gains when improvement rate is high
                self.a = min(self.a * 1.05, self.max_step_size[0])
                self.c = min(self.c * 1.02, self.max_delta[0])
                action = "INCREASE"
            elif rate < 0.3:
                # Decrease gains when improvement rate is low
                self.a = max(self.a * 0.90, self.min_step_size[0])
                self.c = max(self.c * 0.95, self.min_delta[0])
                action = "DECREASE"
            else:
                action = "NO_CHANGE"

            # Log gain changes
            if self.logging_enabled:
                if action != "NO_CHANGE":
                    self.logger.info(f"ADAPTIVE_GAINS - {action}: rate={rate:.2f}, "
                                f"a: {a_old:.6f} -> {self.a:.6f}, "
                                f"c: {c_old:.6f} -> {self.c:.6f}")
                else:
                    self.logger.debug(f"ADAPTIVE_GAINS - NO_CHANGE: rate={rate:.2f}")


"""
Extremum Seeking SPSA Tuning Guide
===================================

This guide provides practical advice for tuning the SPSA extremum seeking parameters.

## Key Parameters and Their Effects

### 1. Gain Coefficient (a)
- **Purpose**: Controls the step size for parameter updates
- **Range**: 0.001 - 0.1 (for rewards in [-10, 10])
- **Tuning**:
  - Too small: Slow convergence, may get stuck in local optima
  - Too large: Unstable behavior, oscillations around optimum
  - Start with: 0.01 for reward range [-10, 10]

### 2. Perturbation Coefficient (c)
- **Purpose**: Controls exploration magnitude for gradient estimation
- **Range**: 0.01 - 0.1
- **Tuning**:
  - Too small: Poor gradient estimates (noise-dominated)
  - Too large: Overshoots, inaccurate local gradient information
  - Start with: 0.05 for reward range [-10, 10]

### 3. Stabilization Parameter (A)
- **Purpose**: Prevents large gains in early iterations
- **Range**: 10 - 1000
- **Tuning**:
  - Larger A: More conservative early learning
  - Smaller A: Faster initial learning, but more aggressive
  - Start with: 100

### 4. Gain Decay Exponents (α, β)
- **Purpose**: Controls how quickly gains and perturbations decay
- **Typical Values**: α = 0.602, β = 0.101 (theoretically optimal)
- **Tuning**:
  - Keep these at theoretical values unless needed otherwise

### 5. Discount Factor (γ)
- **Purpose**: Balances immediate vs. future rewards
- **Range**: 0.9 - 0.99 for typical RL problems
- **Tuning**:
  - Higher γ (0.99): Focus on long-term performance
  - Lower γ (0.9): More responsive to recent rewards

## Tuning Procedure

### Step 1: Start with Conservative Values
```python
a = 0.01      # Conservative gain
c = 0.05      # Moderate perturbation
A = 100       # Standard stabilization
gamma = 0.95  # Balanced discounting
```

### Step 2: Monitor Key Metrics
Watch the logs for:
- **Gradient magnitude**: Should decrease as you approach optimum
- **Step magnitude**: Should gradually decrease over time
- **Reward improvement**: Should show steady upward trend
- **Improvement rate**: Should stabilize around 50% (random walk) at optimum

### Step 3: Adjust Based on Behavior

#### If Convergence is Too Slow:
- Increase `a` by factor of 1.5-2x
- Slightly increase `c` if gradient estimates are noisy
- Reduce `A` for more aggressive early learning

#### If Behavior is Unstable:
- Decrease `a` by factor of 0.5x
- Reduce `c` if perturbations are too large
- Increase `A` for more conservative early learning

#### If Gradient Estimates are Noisy:
- Increase `c` by factor of 1.2-1.5x
- Consider reducing `a` to compensate for larger perturbations

#### If Stuck in Local Optima:
- Temporarily increase `c` for better exploration
- Consider occasional gain resets (increase `a`)

### Step 4: Fine-Tuning for Specific Reward Ranges

#### For Small Rewards ([-1, 1]):
```python
a = 0.001     # Much smaller gains
c = 0.01      # Smaller perturbations
```

#### For Large Rewards ([-100, 100]):
```python
a = 0.1       # Larger gains
c = 0.5       # Larger perturbations
```

## Common Patterns and Solutions

### Oscillations Around Optimum
- **Symptom**: Control signal oscillates, reward fluctuates
- **Solution**: Decrease `a`, increase `A` for more conservative gains

### No Convergence
- **Symptom**: Control signal keeps changing, no reward improvement
- **Solution**: Increase `c` for better gradient estimates, check reward function

### Very Slow Learning
- **Symptom**: Tiny parameter changes, minimal reward change
- **Solution**: Increase `a`, possibly decrease `A` for less damping

### Jumping Between Optima
- **Symptom**: Large parameter jumps, inconsistent rewards
- **Solution**: Decrease `c` and `a`, increase `A`

## Performance Indicators

### Good Convergence:
- Gradient magnitude decreases smoothly
- Step sizes follow theoretical decay rates
- Improvement rate approaches ~50% at optimum
- Reward curve shows monotonic improvement then stabilization

### Problematic Behavior:
- Gradient magnitude increases or stays high
- Step sizes don't decay properly
- Improvement rate stays consistently high (>80%) or low (<20%)
- Highly erratic reward patterns

## Practical Tips

1. **Start Simple**: Use default parameters, observe baseline behavior
2. **Change One Parameter**: Adjust only one parameter at a time
3. **Log Everything**: Use the detailed logging to understand behavior
4. **Be Patient**: SPSA may need hundreds of iterations to converge
5. **Validate**: Test final parameters across different initial conditions
6. **Consider Bounds**: Ensure control signal bounds are appropriate for your system

Remember that tuning is iterative and system-specific. These guidelines provide starting points, but optimal parameters depend on your specific problem dynamics and reward landscape.
"""
