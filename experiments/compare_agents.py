"""
Compare standard SAC agent with n-step lookahead version.
"""

import os
import torch
import gymnasium as gym
import time
import numpy as np
import pandas as pd

from environment import ElectricityMarketEnv, RewardType, DemandType
from agents import SACAgent, SACAgentAhead
from training import train_agent, evaluate, calculate_policy_entropy
from utils import (
    set_seed,
    get_config,
    NormalizedEnv,
    plot_training_history
)

def register_env():
    """Register the electricity market environment."""
    gym.envs.register(
        id="ElectricityMarketEnv-v0",
        entry_point="environment.electricity_market:ElectricityMarketEnv",
    )

def compare_agents(reward_type, demand_type, num_episodes=150, eval_episodes=5, seed=42):
    """
    Compare standard SAC agent with n-step lookahead version for a specific configuration.

    Args:
        reward_type: Type of reward function
        demand_type: Type of demand function
        num_episodes: Number of episodes to train for
        eval_episodes: Number of episodes to evaluate on
        seed: Random seed

    Returns:
        results: Dictionary with comparison results
        training_histories: Dictionary with training rewards
    """
    print(f"\nComparing agents with {reward_type.value} reward and {demand_type.value} demand")

    # Get hyperparameters
    hyperparams = get_config(reward_type, demand_type)
    hyperparams["num_episodes"] = num_episodes

    # Create environments for both agents
    env_standard = gym.make('ElectricityMarketEnv-v0', params=hyperparams)
    env_ahead = gym.make('ElectricityMarketEnv-v0', params=hyperparams)

    # Wrap the environments with the normalizer
    normalized_env_standard = NormalizedEnv(env_standard)
    normalized_env_ahead = NormalizedEnv(env_ahead)

    # Set state dimension and max steps
    hyperparams["state_dim"] = env_standard.observation_space.shape[0]
    hyperparams["battery_capacity"] = env_standard.unwrapped.battery_capacity
    hyperparams["max_steps_per_episode"] = env_standard.unwrapped.max_steps

    # Create n-step hyperparams
    hyperparams_ahead = hyperparams.copy()
    hyperparams_ahead["n_step"] = 10

    # Create agents
    standard_agent = SACAgent(hyperparams)
    ahead_agent = SACAgentAhead(hyperparams_ahead)

    # Train standard agent
    print("\nTraining standard SAC agent...")
    standard_rewards = train_agent(standard_agent, normalized_env_standard, num_episodes, seed=seed)

    # Train ahead agent
    print("\nTraining SAC agent with n-step lookahead...")
    ahead_rewards = train_agent(ahead_agent, normalized_env_ahead, num_episodes, seed=seed)

    # Evaluate standard agent
    print("\nEvaluating standard SAC agent...")
    avg_reward_standard, standard_eval_rewards = evaluate(standard_agent, env_standard, eval_episodes, seed=seed+1000)

    # Evaluate ahead agent
    print("\nEvaluating SAC agent with n-step lookahead...")
    avg_reward_ahead, ahead_eval_rewards = evaluate(ahead_agent, env_ahead, eval_episodes, seed=seed+1000)

    # Calculate metrics
    avg_reward_standard_latter = np.mean(standard_rewards[50:])
    std_reward_standard_latter = np.std(standard_rewards[50:])
    std_reward_standard_eval = np.std(standard_eval_rewards)

    avg_reward_ahead_latter = np.mean(ahead_rewards[50:])
    std_reward_ahead_latter = np.std(ahead_rewards[50:])
    std_reward_ahead_eval = np.std(ahead_eval_rewards)

    # Calculate policy entropy
    standard_entropy = calculate_policy_entropy(standard_agent, env_standard)
    ahead_entropy = calculate_policy_entropy(ahead_agent, env_ahead)

    # Store results
    results = {
        "standard_sac": {
            'agent': 'Standard SAC',
            'avg_reward_latter': avg_reward_standard_latter,
            'std_reward_latter': std_reward_standard_latter,
            'avg_reward_eval': avg_reward_standard,
            'std_reward_eval': std_reward_standard_eval,
            'policy_entropy': standard_entropy,
            'agent_instance': standard_agent,
            'env': env_standard
        },
        "ahead_sac": {
            'agent': 'SAC with n-step',
            'avg_reward_latter': avg_reward_ahead_latter,
            'std_reward_latter': std_reward_ahead_latter,
            'avg_reward_eval': avg_reward_ahead,
            'std_reward_eval': std_reward_ahead_eval,
            'policy_entropy': ahead_entropy,
            'agent_instance': ahead_agent,
            'env': env_ahead
        }
    }

    training_histories = {
        "standard_sac": standard_rewards,
        "ahead_sac": ahead_rewards
    }

    # Print improvement percentage
    improvement = (avg_reward_ahead - avg_reward_standard) / abs(avg_reward_standard) * 100
    print(f"\nPerformance improvement with n-step lookahead: {improvement:.2f}%")

    # Save the agents
    os.makedirs('models', exist_ok=True)
    standard_agent.save(f"models/standard_sac_{reward_type.value}_{demand_type.value}.pt")
    ahead_agent.save(f"models/ahead_sac_{reward_type.value}_{demand_type.value}.pt")

    return results, training_histories

def compare_all_configurations(num_episodes=150, eval_episodes=5, seed=42):
    """
    Compare standard SAC and n-step SAC for multiple configurations.

    Args:
        num_episodes: Number of episodes to train for
        eval_episodes: Number of episodes to evaluate on
        seed: Random seed
    """
    set_seed(seed)
    register_env()

    # Define configurations to test
    configs = [
        (RewardType.PROFIT, DemandType.GAUSSIAN),
        (RewardType.PROFIT, DemandType.SINUSOIDAL)
    ]

    all_results = {}
    all_training_histories = {}

    for reward_type, demand_type in configs:
        config_name = f"{reward_type.value}_{demand_type.value}"
        results, training_histories = compare_agents(
            reward_type, demand_type, num_episodes, eval_episodes, seed
        )

        all_results[config_name] = results
        all_training_histories[config_name] = training_histories

        # Plot training history for this configuration
        plot_training_history(
            training_histories,
            f"SAC vs n-step SAC ({config_name})"
        )

        # Create and print results DataFrame for this configuration
        results_df = pd.DataFrame([
            {
                'Agent': results[agent_name]['agent'],
                'Avg Reward (Eps 50+)': results[agent_name]['avg_reward_latter'],
                'Std Reward (Eps 50+)': results[agent_name]['std_reward_latter'],
                'Avg Reward (Eval)': results[agent_name]['avg_reward_eval'],
                'Std Reward (Eval)': results[agent_name]['std_reward_eval'],
                'Policy Entropy': results[agent_name]['policy_entropy']
            }
            for agent_name in ['standard_sac', 'ahead_sac']
        ])

        print(f"\nResults for {config_name}:")
        print(results_df.to_string(index=False))

        # Save results to CSV
        results_df.to_csv(f"comparison_{config_name}.csv", index=False)

    return all_results, all_training_histories

if __name__ == "__main__":
    all_results, all_training_histories = compare_all_configurations()