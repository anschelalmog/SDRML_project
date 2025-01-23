import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()

        # Actor network: outputs mean of action distribution
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)  # outputs raw action
        )

        # Critic network: outputs state-value
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        """
        Returns mean_action, state_value
        """
        mean_action = self.actor(state)
        state_value = self.critic(state)
        return mean_action, state_value

class Agent:
    """
    Minimal training logic for an actor-critic agent with continuous actions.
    """
    def __init__(self, config):
        self.config = config
        self.gamma = config["gamma"]
        self.lr = config["lr"]

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]

        self.model = ActorCritic(self.state_dim, self.action_dim, hidden_size=64)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # For collecting transitions in a single episode
        self.clear_buffer()

    def clear_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.values = []
        self.dones = []

    def get_action(self, state):
        """
        Given state, sample action from the actor's output distribution.
        For simplicity here, we'll treat the actor's output as the direct action (deterministic).
        You can add Gaussian noise or a distribution if you prefer.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mean_action, value = self.model(state_t)

        # Here we do a simple "deterministic" policy, you could do normal distribution sampling:
        action = mean_action.detach().numpy()[0]

        # Keep track for training
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value.item())
        return action

    def store_outcome(self, reward, done):
        self.rewards.append(reward)
        self.dones.append(done)

    def finish_episode(self):
        """
        One-step actor-critic style update with advantage = R - V(s).
        (Or we can do multi-step returns. This is minimal for illustration.)
        """
        # Convert buffers to tensors
        rewards_t = torch.FloatTensor(self.rewards)
        values_t = torch.FloatTensor(self.values)
        dones_t = torch.FloatTensor(self.dones)

        # We will do a simple "discounted sum" approach from each step to the end
        returns = []
        G = 0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            if d:
                G = 0  # reset on done
            G = r + self.gamma * G
            returns.insert(0, G)
        returns_t = torch.FloatTensor(returns)

        # Advantage
        advantages = returns_t - values_t

        # Re-run forward pass to get mean_action + values (for logprob & such)
        states_t = torch.FloatTensor(self.states)
        mean_action_batch, value_batch = self.model(states_t)

        # Critic loss: MSE of advantage
        critic_loss = advantages.pow(2).mean()

        # Actor loss: for deterministic policy, negative advantage as "loss"
        # (In a real algorithm, you'd handle log-likelihood or distribution.)
        # We'll just treat action = mean, so we can't do a real gradient step w.r.t. action selection.
        # This is purely for demonstration, so let's do a pseudo-loss:
        actor_loss = -advantages.mean()

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffers
        self.clear_buffer()
