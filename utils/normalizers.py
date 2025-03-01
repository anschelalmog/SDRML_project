import numpy as np

class NormalizedEnv:
    """
    Wrapper for environment normalization to improve learning stability.

    Normalizes observations and scales rewards.

    Attributes:
        env: The environment to wrap
        state_mean: Running mean of states
        state_std: Running standard deviation of states
        alpha: Update rate for running statistics
        reward_scale: Scale factor for rewards
    """

    def __init__(self, env, reward_scale=0.1):
        """
        Initialize the normalized environment wrapper.

        Args:
            env: The environment to wrap
            reward_scale: Scale factor for rewards
        """
        self.env = env
        # For state normalization
        self.state_mean = np.zeros(3)  # [SoC, Demand, Price]
        self.state_std = np.ones(3)
        self.alpha = 0.001  # Update rate for running statistics
        # For reward scaling
        self.reward_scale = reward_scale

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Options for resetting

        Returns:
            state: Normalized initial state
            info: Information from the environment
        """
        state, info = self.env.reset(seed=seed, options=options)
        return self._normalize_state(state), info

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            next_state: Normalized next state
            reward: Scaled reward
            terminated: Whether the episode has terminated
            truncated: Whether the episode has been truncated
            info: Information from the environment
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self._update_normalization(next_state)
        scaled_reward = reward * self.reward_scale
        return self._normalize_state(next_state), scaled_reward, terminated, truncated, info

    def _normalize_state(self, state):
        """
        Normalize a state to improve learning stability.

        Args:
            state: State to normalize

        Returns:
            normalized_state: Normalized state
        """
        return (state - self.state_mean) / (self.state_std + 1e-8)

    def _update_normalization(self, state):
        """
        Update running mean and standard deviation for state normalization.

        Args:
            state: State to update with
        """
        self.state_mean = (1 - self.alpha) * self.state_mean + self.alpha * state
        self.state_std = (1 - self.alpha) * self.state_std + self.alpha * np.abs(state - self.state_mean)


def normalize_array(array):
    """
    Normalize an array to the range [0, 1].

    Args:
        array: Array to normalize

    Returns:
        normalized_array: Normalized array
    """
    min_val = np.min(array)
    max_val = np.max(array)
    return (array - min_val) / (max_val - min_val)