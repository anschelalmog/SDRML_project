import argparse
import yaml
import os
import torch
import random
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import subprocess

def launch_tensorboard(logdir='runs', port=6006):
    """
    Launch TensorBoard from within the code.

    Args:
        logdir (str): Directory where TensorBoard logs are stored.
        port (int): Port to run TensorBoard on.
    """
    try:
        print(f"Launching TensorBoard at http://localhost:{port}")
        subprocess.Popen(['tensorboard', f'--logdir={logdir}', f'--port={port}'])
    except FileNotFoundError:
        print("TensorBoard is not installed. Install it with 'pip install tensorboard'.")


def parse_args():
    parser = argparse.ArgumentParser(description="Electricity Market RL Agent")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file.')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    return parser.parse_args()


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

    writer = SummaryWriter(log_dir=log_dir)
    launch_tensorboard(logdir=log_dir)

    args.device = device
    args.run_id = run_id
    args.log_dir = log_dir
    args.writer = writer

    return args
