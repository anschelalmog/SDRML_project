import pytest
import numpy as np
from argparse import Namespace

# Adjust the import paths based on your project structure.
from rl_project.environment import ElectricityMarketEnv


@pytest.fixture
def dummy_args():
    args = Namespace()
    args.episodes = 10
    args.gamma = 0.99
    args.tau = 0.005
    args.batch_size = 16  # Lower batch size for faster tests.
    args.actor_lr = 3e-4
    args.critic_lr = 3e-4
    args.buffer_capacity = 1000
    args.automatic_entropy_tuning = True
    args.reward_type = "profit"
    args.device = "cpu"
    args.eval_episodes = 2
    return args


@pytest.fixture
def env(dummy_args):
    env = ElectricityMarketEnv(args=dummy_args)
    env.seed(0)
    return env


class TestElectricityMarketEnv:
    def test_reset_observation(self, env):
        obs = env.reset()
        assert obs.shape == (3,), "Observation should be 3-dimensional"
        assert 0 <= obs[0] <= env.battery_capacity
        assert obs[1] >= 0, "Demand should be non-negative"
        assert obs[2] >= 0, "Price should be non-negative"

    def test_step_charge(self, env):
        env.reset()
        env.soc = 50.0
        action = np.array([20.0], dtype=np.float32)
        obs, reward, done, info = env.step(action)[:4]
        expected_charge = min(20.0, env.battery_capacity - 50.0)
        assert np.isclose(env.soc, 50.0 + expected_charge), "SoC should increase by the charged amount"
        assert reward == 0.0, "Charging should yield zero reward"

    def test_step_discharge(self, env):
        env.reset()
        env.soc = 80.0
        env._demand_function = lambda t: 10.0
        env._price_function = lambda t: 50.0
        action = np.array([-30.0], dtype=np.float32)
        obs, reward, done, info = env.step(action)[:4]
        expected_reward = 20.0 * 50.0
        expected_soc = 80.0 - 30.0
        assert np.isclose(env.soc, expected_soc), "SoC should decrease by the discharged amount"
        assert np.isclose(reward, expected_reward), "Reward should equal sold_energy * price"

    def test_step_invalid_action(self, env):
        env.reset()
        action = np.array([200.0], dtype=np.float32)
        obs, reward, done, info = env.step(action)[:4]
        expected_charge = min(200.0, env.battery_capacity - 50.0)
        assert np.isclose(env.soc, 50.0 + expected_charge), "Action should be clipped to battery capacity"

    def test_episode_termination(self, env):
        env.reset()
        steps = 0
        done = False
        while not done:
            action = np.array([0.0], dtype=np.float32)
            result = env.step(action)
            done = result[2]
            steps += 1
        assert steps == env.max_steps, "Episode should terminate after max_steps"

    def test_observation_bounds(self, env):
        env.reset()
        for _ in range(10):
            action = np.array([0.0], dtype=np.float32)
            obs, reward, done, info = env.step(action)[:4]
            assert env.observation_space.contains(obs), "Observation is out of bounds"
