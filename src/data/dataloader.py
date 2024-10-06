import torch
from torch.utils.data import DataLoader, Dataset


def collate_fn(batch):
    clean_files, recorded_files, recorded_paths = zip(*batch)

    # Find maximum file length and pad all tensors in batch to that length
    max_length = max([file.shape[1] for file in clean_files])
    clean_files = [
        torch.nn.functional.pad(file, (0, max_length - file.shape[1]))
        for file in clean_files
    ]
    recorded_files = [
        torch.nn.functional.pad(file, (0, max_length - file.shape[1]))
        for file in recorded_files
    ]

    clean_files = torch.stack(clean_files)
    recorded_files = torch.stack(recorded_files)

    return clean_files, recorded_files, recorded_paths


def get_loader(dataset: Dataset, batch_size: int, device: str) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8 if device != "cpu" else 0,
        pin_memory=True
    )
