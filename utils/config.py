"""
Configuration and hyperparameters for the electricity market RL project.
"""

import torch
import random
import numpy as np
from environment.reward_types import RewardType, DemandType

def set_seed(seed=42):
    """
    Set the random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to {seed}")
    return seed


# Default hyperparameters for SAC
BASE_HYPERPARAMS = {
    "battery_capacity": 100,
    "initial_soc": 50.0,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,
    "automatic_entropy_tuning": True,
    "hidden_dim": 128,
    "actor_lr": 3e-4,
    "critic_lr": 3e-4,
    "alpha_lr": 3e-4,
    "replay_capacity": 100_000,
    "batch_size": 256,
    "num_episodes": 150,
    "max_steps": 200,
    "updates_per_step": 2
}


def get_config(reward_type=RewardType.PROFIT, demand_type=DemandType.GAUSSIAN, use_n_step=False):
    """
    Get configuration hyperparameters for a specific experiment.

    Args:
        reward_type: Type of reward function to use
        demand_type: Type of demand function to use
        use_n_step: Whether to use n-step returns

    Returns:
        hyperparams: Dictionary of hyperparameters
    """
    hyperparams = BASE_HYPERPARAMS.copy()
    hyperparams["reward type"] = reward_type
    hyperparams["demand type"] = demand_type

    if use_n_step:
        hyperparams["n_step"] = 10

    return hyperparams