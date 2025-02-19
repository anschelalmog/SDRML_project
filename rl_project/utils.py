import argparse
import yaml
import os
import torch
import random
import numpy as np
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
import subprocess
import sys
from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
from loguru import logger

# def launch_tensorboard(logdir='runs', port=6006):
#     """
#     Launch TensorBoard from within the code.
#
#     Args:
#         logdir (str): Directory where TensorBoard logs are stored.
#         port (int): Port to run TensorBoard on.
#     """
#     try:
#         print(f"Launching TensorBoard at http://localhost:{port}")
#         subprocess.Popen(['tensorboard', f'--logdir={logdir}', f'--port={port}'])
#     except FileNotFoundError:
#         print("TensorBoard is not installed. Install it with 'pip install tensorboard'.")


runtime_log_filename = datetime.now().strftime("logs/runtime_%d_%m_%Y_%H_%M.log")
logger.remove()  # Remove default handler
# logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
logger.add(runtime_log_filename, rotation="10MB", level="TRACE")  # Log to file

def get_logger():
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="Electricity Market RL Agent")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file.')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    args, unknown = parser.parse_known_args()
    return args


def set_run(args):
    # Load configuration from YAML
    # with open(args.config, 'r') as file:
    #     config = yaml.safe_load(file)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Run ID and logging
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join('out/', run_id)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging results to: {log_dir}")

    # writer = SummaryWriter(log_dir=log_dir)
    # launch_tensorboard(logdir=log_dir)

    args.device = device
    args.run_id = run_id
    args.log_dir = log_dir
    # args.writer = writer


    return args


def register_env():
    """
    Register the custom ElectricityMarket environment with Gym.
    """
    register(
        id='ElectricityMarket-v0',
        entry_point='rl_project.environment:ElectricityMarketEnv',  # Path to your environment class
        kwargs={'args': None},  # Optional arguments to pass to the environment
        max_episode_steps=200  # Default number of steps per episode
    )

class MetricsLogger:
    def __init__(self):
        """
        A simple class to store and plot training metrics.
        """
        self.episode_rewards = []      # store total reward each episode
        self.q_losses = []             # store critic (Q) loss at each update
        self.policy_losses = []        # store actor (policy) loss at each update

    def add_episode_reward(self, reward):
        """Store the reward obtained in a single episode."""
        self.episode_rewards.append(reward)

    def add_losses(self, q_loss, policy_loss):
        """Store the losses from an Agent update step."""
        self.q_losses.append(q_loss)
        self.policy_losses.append(policy_loss)

    def plot(self, show: bool = True, save_path: str = None):
        """
        Plot the stored metrics. If show=True, display on screen.
        If save_path is provided, saves the figure to that path.
        """
        fig, axes = plt.subplots(3, 1, figsize=(8, 10))
        # Episode Rewards
        axes[0].plot(self.episode_rewards, label='Episode Reward', color='blue')
        axes[0].set_title('Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].legend()

        # Q-Loss
        axes[1].plot(self.q_losses, label='Q-Loss', color='green')
        axes[1].set_title('Critic (Q) Loss')
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Loss')
        axes[1].legend()

        # Policy Loss
        axes[2].plot(self.policy_losses, label='Policy Loss', color='red')
        axes[2].set_title('Actor (Policy) Loss')
        axes[2].set_xlabel('Update Step')
        axes[2].set_ylabel('Loss')
        axes[2].legend()

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()

