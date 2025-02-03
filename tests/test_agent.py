import pytest
import numpy as np
import torch
from argparse import Namespace

# Adjust the import paths based on your project structure.
from rl_project.environment import ElectricityMarketEnv
from rl_project.agent import Agent


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
    args.device = torch.device("cpu")
    args.eval_episodes = 2
    return args


@pytest.fixture
def env(dummy_args):
    env = ElectricityMarketEnv(args=dummy_args)
    env.seed(0)
    return env


@pytest.fixture
def agent(dummy_args, env):
    return Agent(dummy_args, env)


class TestMarketPlayer:
    def test_select_action_bounds(self, agent, env):
        state = env.reset()
        action = agent.select_action(state, evaluate=True)
        assert action.shape == (env.action_space.shape[0],)
        assert np.all(action >= env.action_space.low), "Action is below the lower bound"
        assert np.all(action <= env.action_space.high), "Action is above the upper bound"

    def test_store_transition_and_replay_buffer(self, agent, env):
        initial_len = len(agent.replay_buffer)
        state = env.reset()
        action = agent.select_action(state, evaluate=False)
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward, next_state, float(done))
        new_len = len(agent.replay_buffer)
        assert new_len == initial_len + 1, "Replay buffer length should increase by 1"

    def test_update_no_data(self, agent):
        agent.replay_buffer.buffer = []
        update_result = agent.update()
        assert update_result is None, "Update should return None with an empty replay buffer"

    def test_update_with_data(self, agent, env):
        state = env.reset()
        for _ in range(agent.batch_size):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, float(done))
            state = next_state
            if done:
                state = env.reset()
        update_result = agent.update()
        assert update_result is not None, "Update should return loss values when there is enough data"
        actor_loss, critic_loss = update_result
        assert isinstance(actor_loss, float), "Actor loss should be a float"
        assert isinstance(critic_loss, float), "Critic loss should be a float"

    def test_deterministic_vs_stochastic_action(self, agent, env):
        state = env.reset()
        action_eval1 = agent.select_action(state, evaluate=True)
        action_eval2 = agent.select_action(state, evaluate=True)
        np.testing.assert_allclose(action_eval1, action_eval2, err_msg="Evaluation actions must be deterministic")
        action_train1 = agent.select_action(state, evaluate=False)
        action_train2 = agent.select_action(state, evaluate=False)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(action_train1, action_train2, err_msg="Training actions should be stochastic")

    def test_replay_buffer_sampling(self, agent, env):
        state = env.reset()
        for _ in range(agent.batch_size):
            action = agent.select_action(state, evaluate=False)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, float(done))
            state = next_state
            if done:
                state = env.reset()
        s, a, r, ns, d = agent.replay_buffer.sample(agent.batch_size)
        assert s.shape[0] == agent.batch_size, "States batch size mismatch"
        assert a.shape[0] == agent.batch_size, "Actions batch size mismatch"
        assert r.shape[0] == agent.batch_size, "Rewards batch size mismatch"
        assert ns.shape[0] == agent.batch_size, "Next states batch size mismatch"
        assert d.shape[0] == agent.batch_size, "Done flags batch size mismatch"
