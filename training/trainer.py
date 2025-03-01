import numpy as np
import time
import torch

class Trainer:
    """
    Trainer class for reinforcement learning agents.

    Manages the interaction between the agent and the environment,
    collects experience, and updates the agent's policy.

    Attributes:
        env: The environment to train in
        agent: The agent to train
        num_episodes: Number of episodes to train for
        max_steps_per_episode: Maximum number of steps per episode
        hyperparams: Dictionary of hyperparameters
    """

    def __init__(self, env, agent, hyperparams):
        """
        Initialize the trainer.

        Args:
            env: The environment to train in
            agent: The agent to train
            hyperparams: Dictionary of hyperparameters
        """
        self.env = env
        self.agent = agent
        self.num_episodes = hyperparams.get("num_episodes", 300)
        self.max_steps_per_episode = hyperparams.get("max_steps_per_episode", 200)
        self.hyperparams = hyperparams

    def train(self):
        """
        Train the agent in the environment.

        Returns:
            all_rewards: List of total rewards per episode
        """
        all_rewards = []
        start_time = time.time()

        for ep in range(self.num_episodes):
            state, _ = self.env.reset(seed=ep)  # Reset environment
            total_reward = 0.0

            for step in range(self.max_steps_per_episode):
                self.agent.total_steps += 1

                # Select an action
                action = self.agent.select_action(state)

                # Ensure action is in correct format for environment
                if not isinstance(action, np.ndarray):
                    action = np.array([action], dtype=np.float32)

                # Step the environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                # Store experience in replay buffer
                self.agent.store_transition(state, action.item(), reward, next_state, done)

                # Update current state
                state = next_state

                # Train the agent
                loss = self.agent.learn_step()
                if loss is not None:
                    self.agent.losses.append(loss)
                    self.agent.writer.add_scalar("Loss/step", loss, self.agent.total_steps)

                if done:
                    break

            # Log episode rewards
            self.agent.episode_rewards.append(total_reward)
            self.agent.writer.add_scalar("Reward/episode", total_reward, ep)

            # Print progress every 10 episodes
            if (ep + 1) % 10 == 0 or ep == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode {ep+1}/{self.num_episodes} - Total Reward: {total_reward:.2f} - Elapsed Time: {elapsed_time:.2f}s")

            all_rewards.append(total_reward)

        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")

        return all_rewards


def train_agent(agent, env, num_episodes, seed=None):
    """
    Train an agent in the given environment.

    A simplified training function for use in experiments.

    Args:
        agent: The agent to train
        env: The environment to train in
        num_episodes: Number of episodes to train for
        seed: Random seed

    Returns:
        episode_rewards: List of episode rewards
    """
    episode_rewards = []

    start_time = time.time()
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed+episode if seed is not None else None)
        episode_reward = 0
        done = False
        truncated = False

        # Episode loop
        while not (done or truncated):
            # Select action
            action = agent.select_action(state)
            action = np.array([action], dtype=np.float32)

            # Execute action
            next_state, reward, done, truncated, _ = env.step(action)

            # Store transition
            agent.store_transition(state, action[0], reward, next_state, done or truncated)

            # Update parameters
            if len(agent.replay_buffer) >= agent.batch_size:
                for _ in range(agent.updates_per_step):
                    agent.update_parameters()

            # Update state and reward
            state = next_state
            episode_reward += reward

        # Track episode reward
        episode_rewards.append(episode_reward)

        # Status update every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode+1}/{num_episodes} | Avg Reward (last 10): {avg_reward:.2f}")

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    return episode_rewards