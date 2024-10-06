import os
from typing import List

import torch
from torch import nn

from src.data.files import get_input_dir
from src.util.consts import MAX_LAG


def get_lag(clean_signal: torch.Tensor, recorded_signal: torch.Tensor):
    clean_signal = clean_signal.unsqueeze(0)
    recorded_signal = recorded_signal.unsqueeze(0)

    cross_correlation = torch.nn.functional.conv1d(
        clean_signal, recorded_signal, padding=MAX_LAG
    )

    return torch.argmax(cross_correlation).item() - MAX_LAG


def align_signals(
    clean_signal: torch.Tensor, recorded_signal: torch.Tensor, recorded_filename: str
):
    lag = load_lag(recorded_filename)
    assert lag <= 0, "Recorded signal is ahead of clean signal"

    # Pad clean signal in the beginning and cut the end to align them
    clean_signal = nn.functional.pad(clean_signal, (-lag, 0))
    clean_signal = clean_signal[:, : recorded_signal.shape[1]]

    return clean_signal, recorded_signal


def store_lags(filenames: List[str], lags: List[int]):
    """
    Store the lags for the given filenames in the lags.txt file
    """
    with open(get_input_dir("lags.txt"), "w") as f:
        for filename, lag in zip(filenames, lags):
            f.write(f"{filename}:{lag}\n")


def compute_lags(
    clean_signals: List[torch.Tensor],
    recorded_signals: List[torch.Tensor],
    recorded_filenames: List[str],
    device: str,
):
    """
    Compute the lags for the given signals and store them in the lags.txt file
    """
    lags = [
        get_lag(clean.to(device), recorded.to(device))
        for clean, recorded in zip(clean_signals, recorded_signals)
    ]
    store_lags(lags, recorded_filenames)
    return lags


def load_lag(file_path: str):
    """
    Load the lag for the given filename from the lags.txt file
    """
    filename = os.path.basename(file_path)

    with open(get_input_dir("lags.txt"), "r") as f:
        for line in f:
            if filename in line:
                return int(line.split(":")[0])

    raise ValueError(f"Filename {filename} not found in lags.txt")
