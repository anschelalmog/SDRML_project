"""
Visualize the best performing agent on the electricity market environment.
"""

import os
import torch
import gymnasium as gym
import numpy as np
import pandas as pd

from environment import ElectricityMarketEnv, RewardType, DemandType
from agents import SACAgent, SACAgentAhead
from utils import set_seed, get_config, visualize_episode, plot_demand_functions

def register_env():
    """Register the electricity market environment."""
    gym.envs.register(
        id="ElectricityMarketEnv-v0",
        entry_point="environment.electricity_market:ElectricityMarketEnv",
    )

def load_agent(path, config, use_n_step=False):
    """
    Load a trained agent from a checkpoint.

    Args:
        path: Path to the checkpoint
        config: Configuration dictionary
        use_n_step: Whether to use the n-step agent

    Returns:
        agent: Loaded agent
    """
    if use_n_step:
        agent = SACAgentAhead(config)
    else:
        agent = SACAgent(config)

    agent.load(path)
    return agent

def visualize_best_agent(results_path="sac_experiment_results.csv", seed=42):
    """
    Visualize the best performing agent based on evaluation reward.

    Args:
        results_path: Path to the results CSV file
        seed: Random seed
    """
    set_seed(seed)
    register_env()

    # Load results
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        print(f"Results file {results_path} not found.")
        print("You need to run experiments first. Use experiments/run_sac.py")
        return

    # Find best configuration
    best_idx = results_df['Avg Reward (Eval)'].idxmax()
    best_config = results_df.iloc[best_idx]

    reward_type = RewardType(best_config['Reward Type'])
    demand_type = DemandType(best_config['Demand Type'])

    print(f"Best configuration: Reward={reward_type.value}, Demand={demand_type.value}")
    print(f"Average evaluation reward: {best_config['Avg Reward (Eval)']:.2f}")

    # Create environment with best configuration
    hyperparams = get_config(reward_type, demand_type)
    env = gym.make('ElectricityMarketEnv-v0', params=hyperparams)

    # Load best agent
    model_path = f"models/sac_{reward_type.value}_{demand_type.value}.pt"
    if os.path.exists(model_path):
        config = get_config(reward_type, demand_type)
        config["state_dim"] = env.observation_space.shape[0]
        config["battery_capacity"] = env.unwrapped.battery_capacity

        agent = load_agent(model_path, config)

        # Visualize an episode
        title = f"Best Agent ({reward_type.value}-{demand_type.value})"
        total_reward = visualize_episode(agent, env, title, seed)
        print(f"Visualization episode reward: {total_reward:.2f}")
    else:
        print(f"Model file {model_path} not found.")
        print("You need to train models first. Use experiments/run_sac.py")

def visualize_demand_functions():
    """
    Visualize the different demand functions.
    """
    plot_demand_functions()

if __name__ == "__main__":
    visualize_best_agent()
    visualize_demand_functions()