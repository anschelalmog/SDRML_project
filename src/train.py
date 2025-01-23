# final_project/train.py

import wandb
import gym
# If you use stable_baselines3:
# from stable_baselines3 import PPO

from .enviorment import ElectricityMarketEnv
from .utils import plot_results  # Example optional function to plot or do something.


def run_training(config):
    """
    Creates the environment, sets up the RL agent,
    runs training, and returns training results.
    """

    # 1. Create the custom environment
    env = ElectricityMarketEnv(
        battery_capacity=config["environment"]["battery_capacity"],
        demand_params=config["environment"]["demand_params"],
        price_params=config["environment"]["price_params"],
        max_steps=config["environment"]["max_steps"]
    )

    # 2. (Optional) Wrap environment with Monitor or VecEnv if desired
    # env = gym.wrappers.Monitor(env, "./videos", force=True)

    # 3. Initialize your RL model. Example with PPO:
    # model = PPO("MlpPolicy", env, verbose=1, learning_rate=config["training"]["lr"], ...)
    # For a custom agent, you would code your networks in `agent.py` or in this file.

    # model = ...

    # 4. Train the RL agent
    # model.learn(total_timesteps=config["training"]["timesteps"])

    # 5. Evaluate or test the trained model (optional, or put in separate function)
    # test_rewards = evaluate_model(model, env, episodes=10)
    # wandb.log({"test_reward_mean": sum(test_rewards) / len(test_rewards)})

    # 6. Return some summary data
    return {
        "final_profit": 1234.56  # This is just a placeholder
    }

