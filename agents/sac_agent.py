import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer

class SACAgent:
    """
    Soft Actor-Critic (SAC) agent.

    SAC is a model-free, off-policy algorithm that optimizes a maximum entropy objective.
    It uses soft updates for the target networks and automatic temperature tuning.

    Attributes:
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        max_action: Maximum action value
        hidden_dim: Dimension of the hidden layers
        gamma: Discount factor
        tau: Target network update rate
        alpha: Temperature parameter for entropy regularization
        automatic_entropy_tuning: Whether to automatically tune alpha
        actor_lr: Learning rate for the actor
        critic_lr: Learning rate for the critic
        alpha_lr: Learning rate for alpha
        buffer_size: Size of the replay buffer
        batch_size: Batch size for training
        replay_buffer: Buffer to store transitions
        updates_per_step: Number of updates per environment step
        actor: Actor network
        critic_1, critic_2: Critic networks
        critic_1_target, critic_2_target: Target critic networks
        actor_optimizer, critic_1_optimizer, critic_2_optimizer: Optimizers
        target_entropy: Target entropy for automatic tuning
        log_alpha: Log of alpha parameter
        alpha_optimizer: Optimizer for alpha
        writer: TensorBoard writer
        episode_rewards: List of episode rewards
        losses: List of losses
        total_steps: Total number of steps taken
        device: Device to run the networks on
    """

    def __init__(self, hyperparams):
        """
        Initialize the SAC agent.

        Args:
            hyperparams: Dictionary containing hyperparameters
        """
        self.state_dim = hyperparams.get("state_dim", 3)
        self.action_dim = 1
        self.max_action = hyperparams.get("battery_capacity", 100.0)

        self.hidden_dim = hyperparams.get("hidden_dim", 256)

        self.gamma = hyperparams.get("gamma", 0.99)
        self.tau = hyperparams.get("tau", 0.005)  # Target network update rate
        self.alpha = hyperparams.get("alpha", 0.2)  # Temperature parameter
        self.automatic_entropy_tuning = hyperparams.get("automatic_entropy_tuning", True)

        self.actor_lr = hyperparams.get("actor_lr", 3e-4)
        self.critic_lr = hyperparams.get("critic_lr", 3e-4)
        self.alpha_lr = hyperparams.get("alpha_lr", 3e-4)

        self.buffer_size = hyperparams.get("replay_capacity", 100000)
        self.batch_size = hyperparams.get("batch_size", 256)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        self.updates_per_step = hyperparams.get("updates_per_step", 2)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize actor and critics
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.max_action).to(self.device)

        self.critic_1 = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_2 = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_1_target = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_2_target = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)

        # Initialize target networks with same weights
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            # Target entropy is -|A| (e.g., -1 for single dimension)
            self.target_entropy = -self.action_dim
            # Initialize log_alpha parameter
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)

        # For logging and visualization
        self.writer = SummaryWriter(f"runs/sac_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.episode_rewards = []
        self.losses = []
        self.total_steps = 0

    def select_action(self, state, evaluate=False):
        """
        Select an action given the current state.

        Args:
            state: Current state
            evaluate: If True, use deterministic policy (mean)

        Returns:
            action: Selected action
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

        if evaluate:
            # Use deterministic policy for evaluation
            _, _, mean = self.actor(state)
            return mean.detach().cpu().numpy()[0]
        else:
            # Sample from the policy for exploration
            action, _, _ = self.actor.sample(state)
            return action.detach().cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if isinstance(action, np.ndarray) and action.size == 1:
            action = action.item()

        self.replay_buffer.add(state, action, reward, next_state, done)

    def update_parameters(self):
        """
        Update the parameters of the networks.

        Returns:
            critic_loss: Loss of the critic network
            actor_loss: Loss of the actor network
            alpha_loss: Loss of the alpha parameter
        """
        # Sample from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # Update critic networks
        with torch.no_grad():
            # Sample actions from the target policy
            next_actions, next_log_probs, _ = self.actor.sample(next_states)

            q1_next = self.critic_1_target(next_states, next_actions)
            q2_next = self.critic_2_target(next_states, next_actions)

            min_q_next = torch.min(q1_next, q2_next)
            q_target = rewards + (1 - dones) * self.gamma * (min_q_next - self.alpha * next_log_probs)

        # Current Q-values
        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)

        critic_1_loss = F.mse_loss(q1, q_target)
        critic_2_loss = F.mse_loss(q2, q_target)

        # Update critic networks
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_1.parameters(), max_norm=1.0)
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_2.parameters(), max_norm=1.0)
        self.critic_2_optimizer.step()

        # Update actor network
        # Sample actions from the current policy
        new_actions, log_probs, _ = self.actor.sample(states)

        # Compute Q-values for the new actions
        q1_new = self.critic_1(states, new_actions)
        q2_new = self.critic_2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        # Actor loss: maximize expected return with entropy regularization
        actor_loss = (-min_q_new + self.alpha * log_probs).mean()

        # Update actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # Update temperature parameter
        alpha_loss = None
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Update alpha value
            self.alpha = self.log_alpha.exp().item()

        # Update target networks using soft update
        self._soft_update(self.critic_1, self.critic_1_target)
        self._soft_update(self.critic_2, self.critic_2_target)

        critic_loss = (critic_1_loss + critic_2_loss) / 2

        return critic_loss.item(), actor_loss.item(), alpha_loss.item() if alpha_loss is not None else 0

    def _soft_update(self, source, target):
        """
        Soft update of target network parameters.
        θ_target = τ * θ_source + (1 - τ) * θ_target

        Args:
            source: Source network
            target: Target network
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def learn_step(self):
        """
        Perform one step of learning.

        Returns:
            loss: The total loss value, or None if no update was performed
        """
        # update if we have enough samples in the buffer
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Multiple updates per step for faster learning
        total_critic_loss = 0
        total_actor_loss = 0
        total_alpha_loss = 0

        for _ in range(self.updates_per_step):
            critic_loss, actor_loss, alpha_loss = self.update_parameters()
            total_critic_loss += critic_loss
            total_actor_loss += actor_loss
            total_alpha_loss += alpha_loss

        # Average the losses
        if self.updates_per_step > 0:
            total_critic_loss /= self.updates_per_step
            total_actor_loss /= self.updates_per_step
            total_alpha_loss /= self.updates_per_step

        total_loss = total_critic_loss + total_actor_loss + total_alpha_loss

        return total_loss

    def save(self, path):
        """
        Save the agent's networks to the specified path.

        Args:
            path: Path to save the models
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'critic_1_target': self.critic_1_target.state_dict(),
            'critic_2_target': self.critic_2_target.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
        }, path)

    def load(self, path):
        """
        Load the agent's networks from the specified path.

        Args:
            path: Path to load the models from
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.critic_1_target.load_state_dict(checkpoint['critic_1_target'])
        self.critic_2_target.load_state_dict(checkpoint['critic_2_target'])

        if self.automatic_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()

    def close(self):
        """Close the tensorboard writer."""
        self.writer.close()