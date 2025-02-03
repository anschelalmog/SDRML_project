import numpy as np
import matplotlib.pyplot as plt
import time

class Trainer:
    def __init__(self, args):
        """
        Enhanced trainer for the SAC agent.

        Args:
            env: Environment instance.
            agent: SAC agent instance.
            args: Configuration object with attributes such as episodes, eval_interval, etc.
        """

        self.args = args
        self.episodes = args.episodes
        self.eval_interval = getattr(args, 'eval_interval', 10)

        # Metrics for logging and plotting
        self.episode_rewards = []       # Total reward per episode
        self.episode_lengths = []       # Number of steps per episode
        self.episode_actor_losses = []  # Average actor loss per episode
        self.episode_critic_losses = [] # Average critic loss per episode
        self.eval_rewards = []          # Evaluation rewards recorded periodically
        self.alpha_values = []          # Log entropy coefficient (if applicable)
        self.total_steps = 0            # Overall step counter

        # Optional: TensorBoard writer for logging if provided in args
        self.writer = getattr(args, 'writer', None)

    def train(self, env, agent):
        """
        Training loop for the SAC agent.
        """
        self.env = env
        self.agent = agent

        for episode in range(1, self.episodes + 1):
            state = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_steps = 0

            # For accumulating losses within the episode
            actor_loss_sum = 0.0
            critic_loss_sum = 0.0
            loss_updates = 0

            while not done:
                # Select action (with exploration noise)
                action = self.agent.select_action(state, evaluate=False)

                # Step the environment while handling different Gym versions.
                result = self.env.step(action)
                if len(result) == 5:
                    # New Gym API: (next_state, reward, done, truncated, info)
                    next_state, reward, done, truncated, info = result
                    done = done or truncated  # Combine done and truncated
                else:
                    # Older Gym API: (next_state, reward, done, info)
                    next_state, reward, done, info = result

                # Store transition in the replay buffer
                # (Pass each argument separately to match the agent's store_transition signature.)
                self.agent.store_transition(state, action, reward, next_state, float(done))

                # Update the agent and accumulate losses (if update returns metrics)
                update_metrics = self.agent.update()
                if update_metrics is not None:
                    actor_loss, critic_loss = update_metrics
                    actor_loss_sum += actor_loss
                    critic_loss_sum += critic_loss
                    loss_updates += 1

                # Optionally log the alpha (entropy coefficient) if the agent has it
                if hasattr(self.agent, 'alpha'):
                    self.alpha_values.append(self.agent.alpha.item())

                state = next_state
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1

            # Compute average losses for the episode
            avg_actor_loss = actor_loss_sum / loss_updates if loss_updates > 0 else 0.0
            avg_critic_loss = critic_loss_sum / loss_updates if loss_updates > 0 else 0.0

            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            self.episode_actor_losses.append(avg_actor_loss)
            self.episode_critic_losses.append(avg_critic_loss)

            # Log metrics to console and optionally to TensorBoard
            print(f"Episode {episode}/{self.episodes}: Reward = {episode_reward:.2f}, Steps = {episode_steps}, "
                  f"Avg Actor Loss = {avg_actor_loss:.4f}, Avg Critic Loss = {avg_critic_loss:.4f}")
            if self.writer:
                self.writer.add_scalar('Reward/Episode', episode_reward, episode)
                self.writer.add_scalar('Loss/Actor', avg_actor_loss, episode)
                self.writer.add_scalar('Loss/Critic', avg_critic_loss, episode)
                self.writer.add_scalar('Misc/TotalSteps', self.total_steps, episode)

            # Periodically evaluate the agent
            if episode % self.eval_interval == 0:
                avg_eval_reward = evaluate(self, eval_episodes=getattr(self.args, 'eval_episodes', 10))
                self.eval_rewards.append((episode, avg_eval_reward))
                if self.writer:
                    self.writer.add_scalar('Evaluation/AverageReward', avg_eval_reward, episode)

        # After training, plot the collected metrics
        self.plot_metrics()
        return self

    def plot_metrics(self):
        """
        Plot training metrics: episode rewards, episode lengths, actor/critic losses, and evaluation rewards.
        """
        episodes = range(1, self.episodes + 1)

        plt.figure(figsize=(12, 10))

        # Plot Episode Rewards
        plt.subplot(2, 2, 1)
        plt.plot(episodes, self.episode_rewards, label="Episode Reward", color="blue")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Episode Rewards")
        plt.legend()

        # Plot Episode Lengths
        plt.subplot(2, 2, 2)
        plt.plot(episodes, self.episode_lengths, label="Episode Length", color="orange")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.title("Episode Lengths")
        plt.legend()

        # Plot Actor and Critic Losses
        plt.subplot(2, 2, 3)
        plt.plot(episodes, self.episode_actor_losses, label="Avg Actor Loss", color="green")
        plt.plot(episodes, self.episode_critic_losses, label="Avg Critic Loss", color="red")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.title("Actor & Critic Losses")
        plt.legend()

        # Plot Evaluation Rewards if available
        plt.subplot(2, 2, 4)
        if self.eval_rewards:
            eval_episodes, eval_rewards = zip(*self.eval_rewards)
            plt.plot(eval_episodes, eval_rewards, label="Evaluation Reward", color="purple", marker='o')
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Evaluation Rewards")
            plt.legend()

        plt.tight_layout()
        plt.show()


def evaluate(trainer, eval_episodes=10):
    """
    Evaluate the agent in a noise-free setting for a specified number of episodes.

    Args:
        trainer: The Trainer instance containing the environment and agent.
        eval_episodes: Number of evaluation episodes.

    Returns:
        avg_reward: The average episode reward over the evaluation episodes.
    """
    returns = []
    for _ in range(eval_episodes):
        state = trainer.env.reset()
        done = False
        total_reward = 0.0
        while not done:
            # Select a deterministic action (no exploration noise)
            action = trainer.agent.select_action(state, evaluate=True)
            result = trainer.env.step(action)
            if len(result) == 5:
                state, reward, done, truncated, info = result
                done = done or truncated
            else:
                state, reward, done, info = result
            total_reward += reward
        returns.append(total_reward)
    avg_reward = np.mean(returns)
    print(f"Evaluation over {eval_episodes} episodes: Average Reward = {avg_reward:.2f}")
    return avg_reward
