# train.py
import torch
import numpy as np
import wandb
import gym

from .agent import Agent
from .enviorment import BatteryEnv

def train(config):
    # Create environment
    env = BatteryEnv(config)

    # Update config with dimension info for the agent
    config["state_dim"] = env.observation_space.shape[0]
    config["action_dim"] = env.action_space.shape[0]

    agent = Agent(config)

    all_episode_rewards = []

    for episode in range(config["episodes"]):
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.get_action(state)  # get from policy
            next_state, reward, done, _ = env.step(action)

            agent.store_outcome(reward, done)
            episode_reward += reward
            state = next_state

        # End of episode: do one update
        agent.finish_episode()

        # Book-keeping
        all_episode_rewards.append(episode_reward)
        wandb.log({
            "Episode": episode,
            "Episode Reward": episode_reward
        })

    env.close()
    return all_episode_rewards
