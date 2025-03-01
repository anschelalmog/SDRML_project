# Electricity Market RL

A reinforcement learning project for optimizing battery storage in an electricity market environment. This project implements the Soft Actor-Critic (SAC) algorithm and an enhanced version with n-step lookahead for better handling of delayed rewards.

## Try It in Colab

You can run this project directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FadSFIsqCJfg8aXFp8n0Ty2XsFuPMjDz)

## Project Overview

This project explores the application of reinforcement learning to optimize battery storage operations in a dynamic electricity market. We've developed an environment that models the interaction between a household battery system and the electricity grid with stochastic demand and real-time price fluctuations.

Our environment simulates realistic scenarios using three different demand models:
1. Mixture of Gaussians - representing typical residential usage patterns
2. Sinusoidal - capturing cyclical demand variations throughout the day
3. Step function - modeling discrete jumps in demand at specific times

We compare two reward formulations:
- **Profit**: Rewards the agent solely based on revenue generated from selling surplus energy
- **Internal Demand**: Encourages the agent to prioritize meeting household demand before selling energy

The main contribution is the development of a **Lookahead Critic** using **TD(n)** learning to improve long-term decision-making, which helps mitigate overestimation bias and stabilize training.

![Accumulative Rewards Comparison](assets/accumulative_rewards.png)
*Comparison of reward accumulation across different model configurations*

![Lookahead Critic Performance](assets/lookhaed-gaussian.png)
*Performance comparison of standard SAC vs. SAC with Lookahead Critic*

## Project Structure

```
electricity_market_rl/
│
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
├── main.py                       # Main entry point
│
├── environment/                  # Environment implementation
│   ├── __init__.py
│   ├── electricity_market.py     # Main environment class
│   └── reward_types.py           # Reward function definitions
│
├── agents/                       # Agent implementations
│   ├── __init__.py
│   ├── sac_agent.py              # Base SAC agent
│   ├── sac_agent_ahead.py        # SAC with n-step lookahead
│   ├── networks.py               # Neural network definitions
│   └── replay_buffer.py          # Replay buffer implementations
│
├── training/                     # Training utilities
│   ├── __init__.py
│   ├── trainer.py                # Training loop utilities 
│   └── evaluator.py              # Evaluation utilities
│
├── utils/                        # Utility functions
│   ├── __init__.py
│   ├── config.py                 # Configuration and hyperparameters
│   ├── normalizers.py            # Environment normalizing wrappers
│   └── visualization.py          # Plotting utilities
│
└── experiments/                  # Experiment scripts
    ├── __init__.py
    ├── run_sac.py                # Standard SAC experiments
    ├── compare_agents.py         # Comparison between agent variants
    └── visualize_best.py         # Visualization of best policies
```

## Environment

The **ElectricityMarketEnv** simulates an electricity market with a battery storage system. The agent can charge (positive action) or discharge (negative action) a battery to optimize profit.

### States
- **SoC**: State of Charge - current energy level in the battery
- **D_t**: Household electricity demand at the current timestep
- **P_t**: Market price of electricity at the current timestep

### Action
- A continuous value in **[-battery_cap, battery_cap]**, the amount of charging/discharging the battery

### Reward Types
1. **Profit**: Rewards based solely on selling surplus energy
2. **Internal Demand**: Rewards prioritizing internal demand before selling energy

### Demand Types
1. **Gaussian**: Models demand using two overlapping Gaussian distributions
2. **Sinusoidal**: Implements a combination of sine waves with different frequencies
3. **Step**: Uses discrete jumps in demand at specific time intervals

## Agents

### Soft Actor-Critic (SAC)
A model-free, off-policy algorithm that optimizes a maximum entropy objective. It uses soft updates for the target networks and automatic temperature tuning.

### SAC with n-step Lookahead Critic
An enhanced version of SAC that uses n-step returns for the critic update. This allows the agent to better handle delayed rewards by looking ahead multiple steps, leading to more stable Q-value estimation and improved policy guidance.

The lookahead critic computes Q-value targets using TD(n):
```
y_t = Σ[k=0 to n-1] γ^k r_{t+k} + γ^n [Q(s_{t+n}, a_{t+n}) - α log π(a_{t+n}|s_{t+n})]
```

Where accumulated rewards over n steps are combined with a bootstrapped Q-value at the n-th step.

## Installation

1. Clone the repository
```bash
git clone https://github.com/anschelalmog/SDRML_Project.git
cd SDRML_Project
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

The project can be run using the main.py script with various command-line arguments:

```bash
# Run standard SAC experiments with different configurations
python main.py --experiment sac --episodes 150 --eval-episodes 5 --seed 42

# Compare standard SAC with n-step lookahead SAC
python main.py --experiment compare --episodes 150 --eval-episodes 5 --seed 42

# Visualize the best performing agent
python main.py --experiment visualize

# Visualize demand functions
python main.py --experiment demand
```

## Results

Experiment results are saved in the `results` directory, and trained models are saved in the `models` directory. Visualizations are saved as PNG files in the project root.

Our experiments reveal several insights:
- The lookahead critic substantially improves learning stability and performance, especially in environments with Gaussian demand patterns
- Performance varies across different demand types, with some configurations maximizing profit while others demonstrating more stable behavior
- The approach achieves significantly better cost savings compared to baseline policies

## GitHub Repository

For the complete source code and documentation, visit our [GitHub Repository](https://github.com/anschelalmog/SDRML_Project).

## Authors
- Almog Anschel
- Eden Hindi

## License
This project is licensed under the MIT License - see the LICENSE file for details.