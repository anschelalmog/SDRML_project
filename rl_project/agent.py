import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import random


# --------------------------
# Replay Buffer
# --------------------------
class ReplayBuffer:
    """
    A simple Replay Buffer for storing transitions observed from the environment.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples a batch of transitions.
        Returns:
            state, action, reward, next_state, done: each as a NumPy array.
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# --------------------------
# Actor Network (Policy)
# --------------------------
class Actor(nn.Module):
    """
    Actor network for SAC.
    Given a state, it outputs the mean and log standard deviation of a Gaussian distribution.
    The action is sampled using the reparameterization trick and squashed using tanh.
    """

    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Limits for numerical stability
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state, evaluate=False):
        """
        Sample an action given the state.
        If evaluate is True, returns the deterministic action (mean) after squashing.
        Otherwise, returns a sampled action along with its log probability.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        if evaluate:
            # For evaluation, use the mean (deterministic policy)
            action = torch.tanh(mean) * self.max_action
            log_prob = None
        else:
            # Reparameterization trick
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # sample and add noise
            action = torch.tanh(x_t) * self.max_action

            # Compute log probability and adjust for tanh squashing
            log_prob = normal.log_prob(x_t)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            # Adjustment term for tanh squashing (to correct probability density)
            log_prob -= torch.log(1 - torch.tanh(x_t).pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob


# --------------------------
# Critic Network (Q-function)
# --------------------------
class Critic(nn.Module):
    """
    Critic network for SAC.
    Evaluates the Q-value for a given state and action pair.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # Concatenate state and action along the last dimension
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


# --------------------------
# SAC Agent
# --------------------------
class Agent:
    """
    Soft Actor-Critic Agent.
    This agent encapsulates the actor, two critics, their target networks, optimizers,
    and a replay buffer. It supports automatic entropy tuning for the exploration/exploitation tradeoff.
    """

    def __init__(self, args, env):
        self.device = args.device

        # Hyperparameters (with reasonable default values)
        self.gamma = getattr(args, 'gamma', 0.99)  # Discount factor
        self.tau = getattr(args, 'tau', 0.005)  # Soft update coefficient for target networks
        self.batch_size = getattr(args, 'batch_size', 256)  # Batch size for updates

        # Learning rates
        self.actor_lr = getattr(args, 'actor_lr', 3e-4)
        self.critic_lr = getattr(args, 'critic_lr', 3e-4)

        # Replay Buffer capacity
        self.buffer_capacity = getattr(args, 'buffer_capacity', 1_000_000)

        # Automatic entropy tuning flag and target entropy
        self.automatic_entropy_tuning = getattr(args, 'automatic_entropy_tuning', True)
        self.target_entropy = -env.action_space.shape[0]

        # Environment dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        # Initialize actor and critics (two Q–networks)
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.critic1 = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2 = Critic(self.state_dim, self.action_dim).to(self.device)

        # Create target networks and initialize with critic parameters
        self.critic1_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic2_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()),
                                     lr=self.critic_lr)

        # Entropy coefficient (alpha) with optional automatic tuning
        if self.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.actor_lr)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = getattr(args, 'alpha', 0.2)

        # Replay Buffer for experience storage
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)

    def select_action(self, state, evaluate=False):
        """
        Given a state, select an action according to the current policy.
        If evaluate is True, return a deterministic action.
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action, _ = self.actor.sample(state, evaluate=evaluate)
        return action.detach().cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        Update the networks (actor and critics) using a mini-batch sampled from the replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None  # Not enough samples to update

        # Sample a batch of transitions
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        # --------------------------
        # Critic update
        # --------------------------
        with torch.no_grad():
            # Sample next actions and compute their log probabilities
            next_action, next_log_prob = self.actor.sample(next_state)
            # Compute target Q-values using target networks and take the minimum to mitigate overestimation
            target_Q1 = self.critic1_target(next_state, next_action)
            target_Q2 = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            # Bellman backup for Q functions
            target_value = reward + (1 - done) * self.gamma * target_Q

        # Compute current Q estimates
        current_Q1 = self.critic1(state, action)
        current_Q2 = self.critic2(state, action)
        # Mean Squared Error loss for both critics
        critic_loss = F.mse_loss(current_Q1, target_value) + F.mse_loss(current_Q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --------------------------
        # Actor update
        # --------------------------
        new_action, log_prob = self.actor.sample(state)
        # Evaluate new actions using the current critics
        Q1_new = self.critic1(state, new_action)
        Q2_new = self.critic2(state, new_action)
        Q_new = torch.min(Q1_new, Q2_new)
        # The actor loss maximizes the expected return and entropy
        actor_loss = (self.alpha * log_prob - Q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --------------------------
        # Entropy coefficient update (if automatic tuning is enabled)
        # --------------------------
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

        # --------------------------
        # Soft update of target networks
        # --------------------------
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # For monitoring purposes, you might return losses
        return actor_loss.item(), critic_loss.item()
