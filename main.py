import gym
import rl_project.utils as utl
from rl_project.agent import Agent
from rl_project.trainer import Trainer, evaluate


def main():
    args = utl.parse_args()
    utl.set_run(args)
    utl.register_env()

    # Initialize environment and agent
    electricity_market = gym.make('ElectricityMarket-v0')
    market_player = Agent(args, electricity_market)

    # Train the agent
    trainer = Trainer(args).train(env=electricity_market,
                                  agent=market_player)

    # evaluation
    evaluate(trainer)


if __name__ == "__main__":
    main()
