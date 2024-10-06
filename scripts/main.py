import argparse
import os
import time
from math import ceil

import torch
from torchaudio.functional import deemphasis, preemphasis

from src.data.files import get_file_paths, load_file, write_file
from src.networks.generator import Generator
from src.util.device import set_device
from src.util.signals import get_signal_chunks, patch_signals

tasks = [
    "T1L1",
    "T1L2",
    "T1L3",
    "T1L4",
    "T1L5",
    "T1L6",
    "T1L7",
    "T2L1",
    "T2L2",
    "T2L3",
    "T3L1",
    "T3L2",
]


def main():
    parser = argparse.ArgumentParser(
        description="Enhances files provided in the specified input directory."
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing files to enhance.",
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save enhanced files to.",
    )

    parser.add_argument(
        "task_id",
        type=str,
        help="Task the files belong to.",
    )

    args = parser.parse_args()

    assert os.path.exists(args.input_dir), "Input directory does not exist!"
    assert os.path.exists(args.output_dir), "Output directory does not exist!"
    assert args.task_id in tasks, "Invalid task ID!"

    enhance(args.task_id, args.input_dir, args.output_dir)


def enhance(task_id: str, input_dir: str, output_dir: str):
    start_time = time.time()

    # Load the model
    device = set_device()

    generator_id = 1 if task_id.startswith("T1") else 2
    generator = Generator(device=device)
    generator.load_state_dict(
        torch.load(
            f"../models/generator_{generator_id}.pt",
            weights_only=True,
            map_location=device,
        )
    )

    # Load the files from the input directory
    recorded_paths = get_file_paths("", base_path=input_dir)
    recorded_files = [
        preemphasis(load_file(recorded_path, base_path=input_dir), 0.95)
        for recorded_path in recorded_paths
    ]

    total_audio_length = sum([file.shape[1] for file in recorded_files]) / 16000
    print(f"Loaded {len(recorded_files)} files for task {task_id} from '{input_dir}'.")

    # Split them into chunks for the generator
    recorded_chunks = [get_signal_chunks(file) for file in recorded_files]
    stacked_recorded_chunks = torch.cat(recorded_chunks, dim=0).to(device)

    # Enhance the files in batches of 64
    all_result_chunks = torch.cat(
        [
            generator(stacked_recorded_chunks[i * 64 : (i + 1) * 64])
            for i in range(ceil(stacked_recorded_chunks.shape[0] / 64))
        ],
        dim=0,
    )

    print(f"Enhanced all {all_result_chunks.shape[0]} chunks.")

    for chunks, path in zip(recorded_chunks, recorded_paths):
        result_chunks = all_result_chunks[: len(chunks)]
        all_result_chunks = all_result_chunks[len(chunks) :]

        # Patch the signals back together
        result_file = patch_signals(result_chunks)
        result_file = deemphasis(result_file, 0.95)
        # Save the enhanced files to the output directory
        filename = os.path.basename(path)
        write_file(result_file, filename=filename, base_path=output_dir)

    duration = time.time() - start_time
    rtf = round(duration / total_audio_length, 3)
    print(f"Wrote enhanced files to '{output_dir}' with a RTF of {rtf}.")


if __name__ == "__main__":
    main()
