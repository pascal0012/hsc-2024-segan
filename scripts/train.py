import argparse
import os

from src.segan import SEGAN
from src.util.consts import TASK_1, TASK_2
from src.util.device import set_device


def main():
    parser = argparse.ArgumentParser(
        description="Trains a (Diffusion-)SEGAN model on the provided levels."
    )

    parser.add_argument(
        "--levels",
        type=str,
        nargs="+",
        help="Levels to train on.",
        required=True,
    )

    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=4_000, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--recon_mag",
        type=float,
        default=100,
        help="Magnitude of reconstruction in generator loss",
    )

    parser.add_argument(
        "--diffusion",
        action=argparse.BooleanOptionalAction,
        help="Whether to enable noise on discriminator inputs",
    )

    args = parser.parse_args()

    # Set device
    device = set_device()

    # Decode levels
    if args.levels[0] == "Task_1":
        args.levels = TASK_1
    elif args.levels[0] == "Task_2":
        args.levels = TASK_2
    elif args.levels[0] == "All":
        args.levels = TASK_1 + TASK_2

    segan = SEGAN(
        levels=args.levels,
        hyperparameters={
            "batch_size": args.batch_size,
            "lr": args.lr,
            "l1_mag": args.recon_mag,
            "diffusion": args.diffusion,
        },
        device=device,
    )

    segan.learn(num_episodes=args.epochs)

    segan.write()

    print_results(segan.test())


def print_results(test_results):
    recon_loss, mean_cer, cers, sample_paths, transcriptions = test_results

    print(f"Reconstruction loss: {recon_loss}")
    print(f"Mean CER: {mean_cer}")
    print("----------------------------------")

    for level, cer in cers:
        print(f"CER {level}: {cer}")

    print("----------------------------------")

    for path, transcription in zip(sample_paths, transcriptions):
        print(f"Transcription {os.path.basename(path)}: {transcription}")


if __name__ == "__main__":
    main()
