"""
Run experiments with the standard SAC agent using different reward and demand types.
"""

import os
import torch
import gymnasium as gym
import time
import numpy as np
import pandas as pd

from environment import ElectricityMarketEnv, RewardType, DemandType
from agents import SACAgent
from training import train_agent, evaluate
from utils import (
    set_seed,
    get_config,
    NormalizedEnv,
    plot_training_history,
    create_results_dataframe
)

def register_env():
    """Register the electricity market environment."""
    gym.envs.register(
        id="ElectricityMarketEnv-v0",
        entry_point="environment.electricity_market:ElectricityMarketEnv",
    )

def run_configuration(reward_type, demand_type, num_episodes=150, eval_episodes=5, seed=42):
    """
    Run a configuration with a specific reward and demand type.

    Args:
        reward_type: Type of reward function
        demand_type: Type of demand function
        num_episodes: Number of episodes to train for
        eval_episodes: Number of episodes to evaluate on
        seed: Random seed

    Returns:
        result: Dictionary with experiment results
        rewards: List of rewards per episode
    """
    print(f"\nRunning configuration: Reward={reward_type.value}, Demand={demand_type.value}")

    # Get hyperparameters
    hyperparams = get_config(reward_type, demand_type)
    hyperparams["num_episodes"] = num_episodes

    # Create the environment
    env = gym.make('ElectricityMarketEnv-v0', params=hyperparams)

    # Wrap the environment with the normalizer
    normalized_env = NormalizedEnv(env)

    # Set state dimension and max steps
    hyperparams["state_dim"] = env.observation_space.shape[0]
    hyperparams["battery_capacity"] = env.unwrapped.battery_capacity
    hyperparams["max_steps_per_episode"] = env.unwrapped.max_steps

    # Create agent
    agent = SACAgent(hyperparams)

    # Train agent
    start_time = time.time()
    rewards = train_agent(agent, normalized_env, num_episodes, seed=seed)
    training_time = time.time() - start_time

    # Calculate metrics from latter part of training
    latter_rewards = rewards[50:]
    avg_reward_latter = np.mean(latter_rewards)
    std_reward_latter = np.std(latter_rewards)

    print(f"Performance (eps 50+): Avg reward = {avg_reward_latter:.2f} ± {std_reward_latter:.2f}")

    # Evaluate agent
    avg_reward_eval, eval_rewards = evaluate(agent, env, eval_episodes, seed=seed+1000)
    std_reward_eval = np.std(eval_rewards)

    # Create result dictionary
    result = {
        'agent': 'SAC',
        'reward_type': reward_type.value,
        'demand_type': demand_type.value,
        'avg_reward_latter': avg_reward_latter,
        'std_reward_latter': std_reward_latter,
        'avg_reward_eval': avg_reward_eval,
        'std_reward_eval': std_reward_eval,
        'training_time': training_time,
        'final_entropy': agent.alpha if hasattr(agent, 'alpha') else None,
        'agent_instance': agent,
        'env': env
    }

    # Save the trained agent
    os.makedirs('models', exist_ok=True)
    agent.save(f"models/sac_{reward_type.value}_{demand_type.value}.pt")

    return result, rewards

def run_all_configurations(num_episodes=150, eval_episodes=3, seed=42):
    """
    Run all configurations with different reward and demand types.

    Args:
        num_episodes: Number of episodes to train for
        eval_episodes: Number of episodes to evaluate on
        seed: Random seed

    Returns:
        results: Dictionary mapping configuration names to results
        training_histories: Dictionary mapping configuration names to reward histories
    """
    set_seed(seed)
    register_env()

    # Define demand types to test
    demand_types = [DemandType.GAUSSIAN, DemandType.SINUSOIDAL, DemandType.STEP]

    results = {}
    training_histories = {}

    # Run profit reward with all demand types
    for demand_type in demand_types:
        config_name = f"{RewardType.PROFIT.value}_{demand_type.value}"
        results[config_name], training_histories[config_name] = run_configuration(
            RewardType.PROFIT, demand_type, num_episodes, eval_episodes, seed
        )

    # Run one additional configuration with INTERNAL_DEMAND
    config_name = f"{RewardType.INTERNAL_DEMAND.value}_{DemandType.GAUSSIAN.value}"
    results[config_name], training_histories[config_name] = run_configuration(
        RewardType.INTERNAL_DEMAND, DemandType.GAUSSIAN, num_episodes, eval_episodes, seed
    )

    # Create and print results DataFrame
    results_df = create_results_dataframe(results)
    print("\nComparative Results:")
    print(results_df.to_string(index=False))

    # Plot training histories
    plot_training_history(training_histories, "SAC Training Reward Comparison")

    return results, training_histories, results_df

if __name__ == "__main__":
    results, training_histories, results_df = run_all_configurations()

    # Save results to CSV
    results_df.to_csv("sac_experiment_results.csv", index=False)