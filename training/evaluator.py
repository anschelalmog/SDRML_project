import numpy as np
import torch

def evaluate(agent, env, n_episodes=10, seed=None):
    """
    Evaluate the agent's performance without exploration.

    Args:
        agent: The agent to evaluate
        env: The environment to evaluate in
        n_episodes: Number of episodes to evaluate
        seed: Random seed for reproducibility

    Returns:
        avg_reward: Average reward per episode
        all_rewards: List of rewards for each episode
    """
    all_rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed+ep if seed is not None else None)
        done = False
        truncated = False
        total_reward = 0

        while not (done or truncated):
            action = agent.select_action(state, evaluate=True)  # Use deterministic policy
            action = np.array([action], dtype=np.float32)
            next_state, reward, done, truncated, _ = env.step(action)

            total_reward += reward
            state = next_state

        all_rewards.append(total_reward)
        print(f"Evaluation episode {ep+1}/{n_episodes} - Reward: {total_reward:.2f}")

    avg_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    print(f"Average evaluation reward: {avg_reward:.2f} ± {std_reward:.2f}")

    return avg_reward, all_rewards


def calculate_policy_entropy(agent, env, num_states=10):
    """
    Calculate the entropy of the policy across multiple states.

    Args:
        agent: Agent with an actor network
        env: Environment to sample states from
        num_states: Number of states to sample

    Returns:
        entropy: Average entropy of the policy
    """
    # Reset environment and collect states
    states = []
    for i in range(num_states):
        state, _ = env.reset(seed=i)
        states.append(state)

    states_tensor = torch.FloatTensor(states).to(agent.device)

    with torch.no_grad():
        _, log_std, _ = agent.actor(states_tensor)
        # Entropy of Gaussian distribution: 0.5 * (log(2*pi*e) + 2*log_std)
        entropy = 0.5 * (torch.log(2 * torch.tensor(np.pi) * torch.exp(1)) + 2 * log_std)
        entropy = entropy.mean().item() * agent.action_dim

    return entropy