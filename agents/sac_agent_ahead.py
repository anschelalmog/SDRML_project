import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from .sac_agent import SACAgent
from .networks import CriticAhead
from .replay_buffer import EnhancedReplayBuffer

class SACAgentAhead(SACAgent):
    """
    SAC agent with n-step lookahead critic.

    Extends the standard SAC agent by replacing the critic with the n-step lookahead critic,
    which provides more accurate Q-value estimation for delayed rewards.

    Attributes:
        n_step: Number of lookahead steps
    """

    def __init__(self, hyperparams):
        """
        Initialize the SAC agent with n-step lookahead critic.

        Args:
            hyperparams: Dictionary containing hyperparameters
        """
        # Initialize base SACAgent properties
        super(SACAgentAhead, self).__init__(hyperparams)

        # Override critics with CriticAhead
        n_step = hyperparams.get("n_step", 10)  # Get n_step from hyperparams or default to 10

        # Replace the critics with CriticAhead
        self.critic_1 = CriticAhead(self.state_dim, self.action_dim, self.hidden_dim, n_step=n_step).to(self.device)
        self.critic_2 = CriticAhead(self.state_dim, self.action_dim, self.hidden_dim, n_step=n_step).to(self.device)
        self.critic_1_target = CriticAhead(self.state_dim, self.action_dim, self.hidden_dim, n_step=n_step).to(self.device)
        self.critic_2_target = CriticAhead(self.state_dim, self.action_dim, self.hidden_dim, n_step=n_step).to(self.device)

        # Initialize target networks with same weights
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Optimizers for the critics
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        # Replace standard replay buffer with enhanced version for n-step returns
        self.replay_buffer = EnhancedReplayBuffer(self.buffer_size)

        # For logging
        self.writer = SummaryWriter(f"runs/sac_ahead_{datetime.now().strftime('%Y%m%d_%H%M%S')}")