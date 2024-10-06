import os
from typing import List, Optional, Tuple

import torch
import torchaudio
from dotenv import load_dotenv

from src.util.consts import SAMPLE_RATE

load_dotenv()


def get_file_paths(path: str, base_path: Optional[str] = None) -> List[str]:
    """
    Returns a list of filenames in the provided directory
    """

    base_path = get_input_dir() if base_path is None else base_path
    file_path = os.path.join(base_path, path)
    filenames = os.listdir(file_path)

    filenames.sort()

    return [
        os.path.join(path, filename)
        for filename in filenames
        if filename.endswith(".wav")
    ]


def load_file(path: str, base_path: Optional[str] = None) -> torch.Tensor:
    """
    Loads a wav file from the given path and trims them to sync files up.
    """
    base_path = get_input_dir() if base_path is None else base_path
    data_path = os.path.join(base_path, path)

    data, sr = torchaudio.load(data_path)
    # Ensure sample rate is as expected
    data = torchaudio.functional.resample(data, orig_freq=sr, new_freq=SAMPLE_RATE)

    return data


def write_files(signals: List[Tuple[torch.Tensor, str]]):
    """
    Writes the provided files to the output directory with their respective filenames
    """
    [write_file(signal, filename) for signal, filename in signals]


def write_file(signal: torch.Tensor, filename: str, base_path: Optional[str] = None):
    """
    Writes the provided file to the output directory with the given filename
    """
    base_path = get_output_dir() if base_path is None else base_path

    output_path = os.path.join(base_path, filename)

    torchaudio.save(output_path, signal.cpu().detach(), SAMPLE_RATE)


def create_dir(directory: str):
    """
    Creates a directory and throws an error if it already exists
    """
    output_dir = get_output_dir()
    dir_path = os.path.join(output_dir, directory)

    os.makedirs(dir_path, exist_ok=False)


def get_input_dir(path: Optional[str] = None) -> str:
    """
    Returns the input directory
    """
    input_dir = os.getenv("DATA_PATH")
    assert (
        input_dir is not None
    ), "DATA_PATH environment variable is not set or does not end in data!"

    if path is not None:
        return os.path.join(input_dir, path)

    return input_dir


def get_output_dir(path: Optional[str] = None) -> str:
    """
    Returns the output directory
    """
    output_dir = os.getenv("OUTPUT_PATH")
    assert output_dir is not None and output_dir.endswith(
        "output"
    ), "OUTPUT_PATH environment variable is not set or does not end in output!"

    if path is not None:
        return os.path.join(output_dir, path)

    return output_dir


def write_model(model: torch.nn.Module, run_name: str, model_name: str):
    """
    Writes the provided model to the output directory with the given filename
    """
    base_path = os.path.join(get_output_dir(), run_name)

    output_path = os.path.join(base_path, f"{model_name}.pt")

    torch.save(model.state_dict(), output_path)
