import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_training_history(training_histories, title="Training Reward Comparison"):
    """
    Plot training histories for multiple agents or configurations.

    Args:
        training_histories: Dictionary mapping configuration names to reward histories
        title: Title for the plot
    """
    plt.figure(figsize=(12, 6))

    for config_name, history in training_histories.items():
        plt.plot(history, label=config_name)

    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()


def visualize_episode(agent, env, title="Episode Visualization", seed=42):
    """
    Run and visualize a single episode using a trained agent.

    Args:
        agent: Trained agent
        env: Environment
        title: Title for the plot
        seed: Random seed for reproducibility
    """
    state, _ = env.reset(seed=seed)
    done = False
    truncated = False

    # Initialize lists to store data
    states = []
    actions = []
    rewards = []

    while not (done or truncated):
        states.append(state)
        action = agent.select_action(state, evaluate=True)
        action = np.array([action], dtype=np.float32)
        actions.append(action[0])

        next_state, reward, done, truncated, _ = env.step(action)
        rewards.append(reward)

        state = next_state

    # Convert to numpy arrays for easier slicing
    states = np.array(states)
    actions = np.array(actions)

    # Create figure with 4 subplots
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Plot SOC
    axs[0].plot(states[:, 0], label='Battery SoC')
    axs[0].set_ylabel('State of Charge')
    axs[0].set_title(f'Battery State of Charge Over Time')
    axs[0].legend()
    axs[0].grid(True)

    # Plot demand
    axs[1].plot(states[:, 1], label='Electricity Demand', color='green')
    axs[1].set_ylabel('Demand')
    axs[1].set_title('Household Electricity Demand Over Time')
    axs[1].legend()
    axs[1].grid(True)

    # Plot price
    axs[2].plot(states[:, 2], label='Electricity Price', color='red')
    axs[2].set_ylabel('Price')
    axs[2].set_title('Electricity Market Price Over Time')
    axs[2].legend()
    axs[2].grid(True)

    # Plot actions
    axs[3].plot(actions, label='Agent Actions', color='purple')
    axs[3].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axs[3].set_ylabel('Action (+ charge, - discharge)')
    axs[3].set_xlabel('Time Step')
    axs[3].set_title('Agent Actions Over Time')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

    return np.sum(rewards)


def plot_demand_functions():
    """
    Plot and compare the different demand functions.
    """
    t_values = np.linspace(0, 1, 1000)

    # Gaussian demand
    gaussian = 80 * np.exp(-((t_values - 0.3) ** 2) / (2 * 0.05**2)) + \
               120 * np.exp(-((t_values - 0.75) ** 2) / (2 * 0.08**2))

    # Sinusoidal demand
    sinusoidal = np.flip(np.maximum(
        80 * np.sin(4 * np.pi * t_values-2) +
        30 * np.sin(2 * np.pi * t_values) + 70,
        0
    ))

    # Step demand
    step = np.where((0.2 <= t_values) & (t_values < 0.35), 100, 0) + \
           np.where((0.7 <= t_values) & (t_values < 0.85), 150, 0)

    # Normalize
    from .normalizers import normalize_array
    gaussian = normalize_array(gaussian)
    sinusoidal = normalize_array(sinusoidal)
    step = normalize_array(step)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(t_values, gaussian, '-', label="Mixture of Gaussians", linewidth=2)
    plt.plot(t_values, sinusoidal, '--', label="Sinusoidal", linewidth=2)
    plt.plot(t_values, step, '-', label="Two-Step", linewidth=2)
    plt.xlabel("Normalized Time (t)")
    plt.ylabel("Electricity Demand")
    plt.title("Demand Function Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("demand_functions.png")
    plt.show()


def create_results_dataframe(results, include_agent=False):
    """
    Create a DataFrame from experiment results.

    Args:
        results: Dictionary of results
        include_agent: Whether to include the agent object in the DataFrame

    Returns:
        df: DataFrame of results
    """
    data = []

    for config_name, result in results.items():
        row = {
            'Configuration': config_name,
            'Reward Type': result['reward_type'],
            'Demand Type': result['demand_type'],
            'Avg Reward (Latter)': result['avg_reward_latter'],
            'Std Reward (Latter)': result['std_reward_latter'],
            'Avg Reward (Eval)': result['avg_reward_eval'],
            'Std Reward (Eval)': result['std_reward_eval'],
            'Training Time (s)': result['training_time'],
            'Final Entropy': result.get('final_entropy', None)
        }

        if include_agent:
            row['Agent'] = result.get('agent_instance', None)
            row['Environment'] = result.get('env', None)

        data.append(row)

    df = pd.DataFrame(data)
    return df