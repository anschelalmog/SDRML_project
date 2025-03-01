"""
Main entry point for the Electricity Market RL project.

This script provides a command-line interface to run various experiments
and visualizations for the electricity market battery control problem.
"""

import argparse
import gymnasium as gym
import os

from utils import set_seed
from experiments import (
    run_all_configurations,
    compare_all_configurations,
    visualize_best_agent,
    visualize_demand_functions
)

def register_env():
    """Register the electricity market environment."""
    gym.envs.register(
        id="ElectricityMarketEnv-v0",
        entry_point="environment.electricity_market:ElectricityMarketEnv",
    )

def main():
    """Main function to parse arguments and run experiments."""
    parser = argparse.ArgumentParser(description='Electricity Market RL Experiments')

    parser.add_argument('--experiment', type=str, default='sac',
                        choices=['sac', 'compare', 'visualize', 'demand'],
                        help='Experiment to run')

    parser.add_argument('--episodes', type=int, default=150,
                        help='Number of episodes for training')

    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Number of episodes for evaluation')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed and register environment
    set_seed(args.seed)
    register_env()

    # Create output directory
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Run selected experiment
    if args.experiment == 'sac':
        print("Running SAC experiments with different configurations...")
        results, training_histories, results_df = run_all_configurations(
            num_episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed
        )
        # Save results
        results_df.to_csv("results/sac_experiment_results.csv", index=False)

    elif args.experiment == 'compare':
        print("Comparing standard SAC with n-step lookahead SAC...")
        all_results, all_training_histories = compare_all_configurations(
            num_episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            seed=args.seed
        )

    elif args.experiment == 'visualize':
        print("Visualizing the best performing agent...")
        visualize_best_agent(
            results_path="results/sac_experiment_results.csv",
            seed=args.seed
        )

    elif args.experiment == 'demand':
        print("Visualizing demand functions...")
        visualize_demand_functions()

    print("Done!")

if __name__ == "__main__":
    main()