import rl_project.utils as utl
from rl_project.environment import ElectricityMarketEnv
from rl_project.agent import Agent
from rl_project.trainer import Trainer, evaluate


def main():
    args = utl.parse_args()
    utl.set_run(args)

    # Initialize environment and agent
    electricity_market = ElectricityMarketEnv(args)
    electricity_player = Agent(args)

    # Train the agent
    trainer = Trainer(env=electricity_market,
                      agent=electricity_player,
                      args=args)

    trainer.train()

    # evaluation
    evaluate(trainer)


if __name__ == "__main__":
    main()
