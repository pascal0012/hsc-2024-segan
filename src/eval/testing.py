import os
from typing import Dict, List, Tuple

import torch
from torch import nn
from torchaudio.functional import deemphasis

from src.data.files import create_dir, get_output_dir, write_file, write_model
from src.eval.evaluate_integration import evaluate_parameters
from src.networks.generator import Generator
from src.util.signals import CombinedChunks, patch_signals


class Tester:
    def __init__(self, run_name: str) -> None:
        self.base_path = run_name
        self.loss = nn.MSELoss()
        self.episode = -1

    def test(
        self,
        generator: Generator,
        paths: List[Tuple[str, str]],
        chunks: List[CombinedChunks],
        episode: int,
        device: str,
        write_all: bool = False,
    ) -> Tuple[float, float, Dict[str, float], List[str], List[str]]:
        n_files = len(chunks)

        self.create_episode_dir(episode)
        # Write generator model to episode directory
        write_model(
            model=generator,
            run_name=self.get_episode_path(),
            model_name=f"episode_{self.episode}_generator",
        )

        chunk_recon_loss = 0
        sample_paths = []
        essential_levels = [
            "task_1_level_1",
            "task_1_level_4",
            "task_1_level_7",
            "task_2_level_1",
        ]

        # Pass all recorded chunks through generator at once
        all_recorded_chunks = [
            recorded_chunk
            for _, recorded_chunks in chunks
            for recorded_chunk in recorded_chunks
        ]
        all_recorded_chunks = torch.stack(all_recorded_chunks, dim=0).to(device)
        all_result_chunks = generator(all_recorded_chunks)

        for (clean_chunks, recorded_chunks), (_, recorded_path) in zip(chunks, paths):
            clean_chunks = clean_chunks.to(device)
            # Get result chunks for current file and remove them from chunk list
            result_chunks = all_result_chunks[: len(recorded_chunks)]
            all_result_chunks = all_result_chunks[len(recorded_chunks) :]

            # Calculate loss for chunks
            chunk_recon_loss += self.loss(result_chunks, clean_chunks) / n_files

            # Calculate loss for patched files
            result_file = patch_signals(result_chunks)

            level = "_".join(os.path.basename(recorded_path).split("_")[:4])
            if (level in essential_levels) or write_all:
                # Regulate high frequencies to original levels
                result_file = deemphasis(result_file, 0.95)

                # Create directory for episode and write file
                write_path = self.get_path(recorded_path)
                write_file(result_file, write_path)

                if level in essential_levels:
                    essential_levels.remove(level)
                    sample_paths.append(write_path)

        mean_cer, cers, transcriptions = evaluate_parameters(
            get_output_dir(self.get_episode_path())
        )

        return (
            chunk_recon_loss,
            mean_cer,
            cers,
            sample_paths,
            transcriptions,
        )

    def create_episode_dir(self, episode: int):
        self.episode = episode
        create_dir(self.get_episode_path())

    def get_episode_path(self) -> str:
        episode = (
            f"{self.episode}"
            if self.episode >= 1000
            else f"0{self.episode}"
            if self.episode >= 100
            else f"00{self.episode}"
            if self.episode >= 10
            else f"000{self.episode}"
        )
        return os.path.join(self.base_path, f"episode_{episode}")

    def get_path(self, filepath: str) -> str:
        """
        Creates the directory for the current episodes and returns the path
        """
        filename = os.path.basename(filepath)
        episode_path = self.get_episode_path()

        return os.path.join(episode_path, filename)
