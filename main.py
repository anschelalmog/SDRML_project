import gymnasium as gym
import rl_project.utils as utl
from rl_project.agent import Agent
from rl_project.trainer import Trainer, evaluate

logger = utl.get_logger()

def main():
    args = utl.parse_args()
    utl.set_run(args)
    utl.register_env()

    logger.info("Initialize electricity market simulation")

    electricity_market = gym.make('ElectricityMarket-v0')
    market_player = Agent(args, electricity_market)

    # Train the agent
    logger.info("Initialize agent training")
    trainer = Trainer(args).train(env=electricity_market,
                                  agent=market_player)
    logger.info("Training completed")

    # evaluation
    logger.info("Starting evaluation")
    evaluate(trainer)
    logger.info("Evaluation finished")


if __name__ == "__main__":
    main()
