# main.py
import yaml
import wandb
import argparse
from src.train import train
from src.utils import set_global_seed

def main():
    # Command-line args (optional)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to main config file")
    parser.add_argument("--wandb_config", type=str, default="wandb_config.yaml", help="Path to W&B config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    try:
        with open(args.wandb_config, "r") as f:
            wandb_config = yaml.safe_load(f)["wandb"]
    except FileNotFoundError:
        raise FileNotFoundError(f"W&B config file not found at {args.wandb_config}.")

    wandb.login(key=wandb_config["api_key"])
    set_global_seed(args.seed)
    wandb.init(
        project=wandb_config["project_name"]  # ,
        #  config=config,
        #  name=wandb_config.get("run_name", "default-run-name"),
    )

    # Merge wandb.config (for easy hyperparameter changes from the UI)
    for k, v in wandb.config.items():
        config[k] = v

    # Start training
    train(config)

if __name__ == "__main__":
    main()
