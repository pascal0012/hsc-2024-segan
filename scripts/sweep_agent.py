import wandb

from src.segan import SEGAN
from src.util.device import set_device


def main():
    # Set device
    device = set_device()

    wandb.init()
    config = wandb.config
    hyperparameters = {
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "l1_mag": config["l1_mag"],
        "diffusion": True,
        "diffuser_C": config["diffusion_C"],
        "diffuser_sigma": config["diffusion_sigma"],
        "diffuser_d_target": config["diffusion_d_target"],
    }

    segan = SEGAN(
        levels=config["levels"],
        hyperparameters=hyperparameters,
        device=device,
    )

    segan.learn(num_episodes=3000, sweep=True)

    print(f"Finished run with hyperparameters: {hyperparameters}")


if __name__ == "__main__":
    main()
