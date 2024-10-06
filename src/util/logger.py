import os
from datetime import datetime
from typing import Any, Dict, List

import dotenv
import wandb

from src.data.files import get_output_dir
from src.util.consts import SAMPLE_RATE

dotenv.load_dotenv()


class Logger:
    def __init__(
        self,
        tasks: List[str],
        hyperparameters: Dict[str, Any],
        tags: List[str] = [],
        sweep: bool = False,
    ) -> None:
        self.wandb_disabled = True if os.getenv("DISABLE_WANDB") == "True" else False
        self.run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        print(f"Starting run {self.run_name}")

        if not self.wandb_disabled:
            if not sweep:
                self.init_wandb(tasks, hyperparameters, tags)
        else:
            print("Weights and Biases logging is disabled!")

    def init_wandb(
        self,
        tasks: List[str],
        hyperparameters: Dict[str, Any],
        tags: List[str] = [],
    ):
        wandb.init(
            # set the wandb project where this run will be logged
            project="hsc-2024",
            # track hyperparameters and run metadata
            config=hyperparameters
            | {
                "tasks": tasks,
                "run_name": self.run_name,
                "tags": tags,
            },
        )

    def log_metrics(
        self,
        episode: int,
        iteration: int,
        lr: float,
        file_recon_loss: float = None,
        chunk_recon_loss: float = None,
        mean_cer: float = None,
        cers: Dict[str, float] = {},
        generator_loss: float = None,
        discriminator_loss: float = None,
        audio_paths: List[str] = [],
        transcriptions: List[str] = None,
    ):
        log_content = {
            "episode": episode,
            "iteration": iteration,
            "lr": lr,
        }

        if generator_loss is not None:
            log_content["generator_loss"] = generator_loss
        if discriminator_loss is not None:
            log_content["discriminator_loss"] = discriminator_loss
        if chunk_recon_loss is not None:
            log_content["chunk_loss"] = chunk_recon_loss
        if file_recon_loss is not None:
            log_content["file_loss"] = file_recon_loss
        if mean_cer is not None:
            log_content["mean_cer"] = mean_cer
    
        # CER for each level is logged
        for task in cers:
            log_content[f"cer_{task}"] = cers[task]

        # Audio clips for all provided paths are logged
        for i, audio_path in enumerate(audio_paths):
            log_content[f"audio_clip_{i}"] = wandb.Audio(
                get_output_dir(audio_path),
                SAMPLE_RATE,
                caption=os.path.basename(audio_path),
            )

        if transcriptions is not None:
            log_content["transcriptions"] = transcriptions[1]

        if not self.wandb_disabled:
            wandb.log(log_content)

        print("----------------------------------------------")
        print(
            "\n".join(
                [
                    f"{key}: {value}"
                    for key, value in log_content.items()
                    if type(value) is float or type(value) is int
                ]
            )
        )

    def finish(self):
        if not self.wandb_disabled:
            wandb.finish()
