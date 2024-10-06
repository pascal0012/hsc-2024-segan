from typing import List, NamedTuple, Tuple, Union

import numpy as np
import torch
from torchaudio.functional import preemphasis

from src.data.files import load_file
from src.data.filesampler import PathPair
from src.util.consts import WINDOW_LENGTH
from src.util.lag import align_signals

PatchedSignal = NamedTuple(
    "PatchedSignal", [("signal", torch.Tensor), ("file_path", str)]
)
Chunks = List[torch.Tensor]
CombinedChunks = Tuple[Chunks, Chunks]


def get_signal_chunks(signal: torch.Tensor) -> torch.Tensor:
    signal_residue = signal.shape[1] % (WINDOW_LENGTH // 2)
    signal = (
        signal
        if signal_residue == 0
        else torch.cat(
            (
                signal,
                torch.zeros(
                    [1, (WINDOW_LENGTH // 2) - signal_residue], device=signal.device
                ),
            ),
            dim=1,
        )
    )

    start = 0
    end = WINDOW_LENGTH
    chunks = torch.empty([0, 1, WINDOW_LENGTH], device=signal.device)
    while end <= signal.shape[1]:
        chunks = torch.cat((chunks, signal[:, start:end].unsqueeze(0)), dim=0)

        start += WINDOW_LENGTH // 2
        end += WINDOW_LENGTH // 2

    return chunks


def get_signal_chunk(signal: torch.Tensor, chunk: int) -> torch.Tensor:
    """
    Loads a wav file from the given path and trims them to sync files up.
    Returns a WINDOW_LENGTH sample chunk of the file if chunk is not None, otherwise the entire file is returned.
    """
    chunks = get_signal_chunks(signal)

    assert (
        chunk < chunks.shape[0]
    ), f"Chunk index {chunk} is out of bounds for the provided signal!"

    return chunks[chunk]


def load_chunks_pair(paths: PathPair):
    signal1, signal2 = load_file_pair(paths=paths)

    return get_signal_chunks(signal1), get_signal_chunks(signal2)


def load_chunks_pair_list(sampled_paths: List[PathPair]) -> List[CombinedChunks]:
    return [load_chunks_pair(paths=path_pair) for path_pair in sampled_paths]


def patch_signals(signal_chunks: torch.Tensor) -> torch.Tensor:
    """
    Patches the provided signals together in the order they were received.
    """
    signal = torch.empty([1, WINDOW_LENGTH])
    for i in range(signal_chunks.shape[0]):
        # Make last 50% of previous signal and first 50% of next signal overlap
        if i == 0:
            signal = signal_chunks[i]
        else:
            overlap = (
                signal[:, -WINDOW_LENGTH // 2 :] / 2
                + signal_chunks[i][:, : WINDOW_LENGTH // 2] / 2
            )

            signal = torch.cat(
                (
                    signal[:, : -WINDOW_LENGTH // 2],
                    overlap,
                    signal_chunks[i][:, WINDOW_LENGTH // 2 :],
                ),
                dim=1,
            )

    return signal


def expand_shape(tensor: torch.Tensor, shape: Union[int, List[int]]):
    """
    Expands the given tensor to the given shape
    """

    if isinstance(shape, int):
        shape = [shape]

    if tensor.dim() != len(shape):
        raise ValueError("Cannot expand tensor to a different number of dimensions!")

    for i in range(tensor.dim()):
        if tensor.shape[i] > shape[i]:
            raise ValueError("Cannot expand tensor to a smaller shape!")

    new_tensor = torch.zeros(shape, device=tensor.device)
    slicing_indices = tuple(slice(0, dim) for dim in tensor.shape)
    new_tensor[slicing_indices] = tensor

    return new_tensor


def stft(signal: torch.Tensor, n_fft: int):
    return torch.stft(
        signal.squeeze(1),
        n_fft=n_fft,
        win_length=n_fft // 2,
        hop_length=n_fft // 4,
        return_complex=True,
        window=torch.hann_window(n_fft // 2, device=signal.device),
        onesided=False,
    )


def istft(signal: torch.Tensor, n_fft: int) -> torch.Tensor:
    return torch.istft(
        signal,
        n_fft=n_fft,
        win_length=n_fft // 2,
        hop_length=n_fft // 4,
        return_complex=True,
        window=torch.hann_window(n_fft // 2, device=signal.device),
        onesided=False,
    ).unsqueeze(1)


def load_file_pair(
    paths: Tuple[str, str], align: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    signal1, signal2 = (
        load_file(paths[0]),
        load_file(paths[1]),
    )

    # Align signals using cross-correlation and cut off the excess
    if align:
        signal1, signal2 = align_signals(signal1, signal2, paths[1])

    # Apply preemphasis filter to signal pair
    signal1, signal2 = preemphasis(signal1, 0.95), preemphasis(signal2, 0.95)

    return signal1, signal2


def get_random_chunk_pair(
    signal1: torch.Tensor, signal2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert signal1.shape == signal2.shape, "Signals must have the same length!"
    assert signal1.shape[1] >= WINDOW_LENGTH, "Signal is too short to extract a chunk!"

    start = np.random.randint(0, signal1.shape[1] - WINDOW_LENGTH)
    end = start + WINDOW_LENGTH

    return signal1[:, start:end], signal2[:, start:end]


def load_random_chunk_pair(
    paths: Tuple[str, str], align: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    signal1, signal2 = load_file_pair(paths, align)

    return get_random_chunk_pair(signal1, signal2)
