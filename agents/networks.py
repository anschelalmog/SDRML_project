import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor network for the SAC algorithm.

    Maps state to action distributions, then samples actions from these distributions.
    Parametrizes a Gaussian policy with mean and log standard deviation.

    Attributes:
        net: Main network layers
        mean: Output layer for mean of action distribution
        log_std: Output layer for log standard deviation of action distribution
        max_action: Maximum action value
        action_dim: Dimension of action space
        log_std_min: Minimum log standard deviation to prevent numerical instability
        log_std_max: Maximum log standard deviation
    """

    def __init__(self, state_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mean and log_std outputs
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.max_action = max_action
        self.action_dim = action_dim

        # Minimum log standard deviation to prevent numerical instability
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        """
        Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            mean: Mean of the action distribution
            log_std: Log standard deviation of the action distribution
            mean_scaled: Scaled mean to fit the action range
        """
        features = self.net(state)

        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        # Scale mean to action range
        mean_scaled = self.max_action * torch.tanh(mean)

        return mean, log_std, mean_scaled

    def sample(self, state):
        """
        Sample an action from the policy distribution.

        Args:
            state: Input state tensor

        Returns:
            action: Sampled action
            log_prob: Log probability of the sampled action
            mean_scaled: Deterministic action (mean)
        """
        mean, log_std, mean_scaled = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = self.max_action * y_t

        # Log probability computation with change of variables
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, mean_scaled


class Critic(nn.Module):
    """
    Critic network for the SAC algorithm.

    Implements Q-function estimation, predicting future returns for state-action pairs.

    Attributes:
        net: Neural network for Q-function approximation
    """

    def __init__(self, state_dim, action_dim, hidden_dim):
        """
        Initialize the critic network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layers
        """
        super(Critic, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """
        Forward pass through the network.

        Args:
            state: Input state tensor
            action: Input action tensor

        Returns:
            q_value: Q-value for the state-action pair
        """
        x = torch.cat([state, action], dim=1)
        q_value = self.net(x)
        return q_value


class CriticAhead(Critic):
    """
    Extension of SAC Critic that uses TD(n) with n-step lookahead.

    Predicts future rewards by accumulating n-step returns and bootstrapping
    from the value at the nth step. Provides more accurate Q-value estimation for
    delayed rewards.
    """

    def __init__(self, state_dim, action_dim, hidden_dim, n_step=10):
        super(CriticAhead, self).__init__(state_dim, action_dim, hidden_dim)
        self.n_step = n_step  # number of lookahead steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update(self, replay_buffer, actor, target_critic, alpha, gamma=0.99, batch_size=256):
        """
        Update the critic using TD(n) with n-step returns.

        Args:
            replay_buffer: Replay buffer from which to sample transitions
            actor: Policy network to compute next actions and log-probs
            target_critic: Target critic network for bootstrapping
            alpha: Entropy temperature coefficient
            gamma: Discount factor
            batch_size: Number of samples to train on

        Returns:
            critic_loss: Value of the critic loss
        """
        # Sample a batch of experiences from replay buffer
        states, actions, _, _, _, indices = replay_buffer.sample(batch_size, return_indices=True)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Initialize returns tensor
        batch_returns = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)

        # Lists for bootstrapping
        bootstrap_states = []
        bootstrap_discounts = []
        bootstrap_indices = []

        # Compute multi-step returns for each sampled index
        for batch_idx, idx in enumerate(indices):
            G = 0.0        # cumulative return
            discount = 1.0  # discount factor
            done_flag = False

            # Traverse up to n_step transitions
            for n in range(self.n_step):
                if idx + n >= len(replay_buffer):
                    done_flag = True
                    break

                reward_n = replay_buffer.rewards[idx + n]
                done_n = replay_buffer.dones[idx + n]
                G += discount * reward_n

                if done_n:
                    done_flag = True
                    break

                discount *= gamma

            batch_returns[batch_idx] = G

            # If episode didn't end, bootstrap from the nth state
            if not done_flag:
                last_transition_index = idx + self.n_step - 1
                bootstrap_state = replay_buffer.next_states[last_transition_index]
                bootstrap_states.append(bootstrap_state)
                bootstrap_discounts.append(discount * gamma)
                bootstrap_indices.append(batch_idx)

        # Compute bootstrapped values
        if len(bootstrap_states) > 0:
            bs_tensor = torch.tensor(bootstrap_states, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                # Sample actions from policy
                actions_bs, log_probs_bs, _ = actor.sample(bs_tensor)

                # Get Q-values from target critics
                q1_bs = target_critic[0](bs_tensor, actions_bs)
                q2_bs = target_critic[1](bs_tensor, actions_bs)
                q_min_bs = torch.min(q1_bs, q2_bs)

                # Apply entropy regularization
                target_values = q_min_bs - alpha * log_probs_bs

            # Add bootstrapped values to returns
            for i, batch_idx in enumerate(bootstrap_indices):
                batch_returns[batch_idx] += bootstrap_discounts[i] * target_values[i]

        # Compute current Q-values
        q1 = self(states, actions)

        # Compute loss
        critic_loss = F.mse_loss(q1, batch_returns)

        return critic_loss.item()