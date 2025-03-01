import random
import numpy as np

class ReplayBuffer:
    """
    Store and manage transitions experienced by the agent.

    Allows for uniform sampling of transitions for off-policy learning.

    Attributes:
        capacity: Maximum capacity of the buffer
        buffer: List to store transitions
        position: Current position in the buffer
    """

    def __init__(self, capacity):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum capacity of the buffer
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Size of the batch

        Returns:
            batch: Batch of transitions (state, action, reward, next_state, done)
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            size: Current size of the buffer
        """
        return len(self.buffer)


class EnhancedReplayBuffer(ReplayBuffer):
    """
    Enhanced replay buffer that stores transitions in separate arrays.

    Allows for sequential access to transitions, which is needed for n-step returns.

    Attributes:
        capacity: Maximum capacity of the buffer
        states: Array of states
        actions: Array of actions
        rewards: Array of rewards
        next_states: Array of next states
        dones: Array of done flags
        position: Current position in the buffer
    """

    def __init__(self, capacity):
        super(EnhancedReplayBuffer, self).__init__(capacity)
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if len(self.buffer) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
            self.buffer.append(None)
        else:
            idx = self.position
            self.states[idx] = state
            self.actions[idx] = action
            self.rewards[idx] = reward
            self.next_states[idx] = next_state
            self.dones[idx] = done

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, return_indices=False):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Size of the batch
            return_indices: Whether to return the indices of the sampled transitions

        Returns:
            batch: Batch of transitions and optionally their indices
        """
        indices = np.random.randint(0, len(self.buffer), size=batch_size)

        states = np.array([self.states[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        next_states = np.array([self.next_states[i] for i in indices])
        dones = np.array([self.dones[i] for i in indices])

        if return_indices:
            return states, actions, rewards, next_states, dones, indices
        return states, actions, rewards, next_states, dones