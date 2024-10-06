import os
from collections import namedtuple
from typing import List, Tuple, Union

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset

from src.data.filesampler import PathPair, remove_file_paths
from src.util.signals import load_file_pair, load_random_chunk_pair

load_dotenv()

FileChunkMetadata = namedtuple(
    "FileChunkMetadata", ["clean_path", "recorded_path", "chunk_index"]
)
TrainChunk = namedtuple(
    "TrainChunk", ["clean_chunk", "recorded_chunk", "recorded_path"]
)


class SpeechData(Dataset):
    def __init__(
        self,
        tasks: Union[str, List[str]],
        ignore_paths: List[PathPair] = [],
        return_chunks: bool = True,
        align: bool = True,
    ):
        self.return_chunks = return_chunks
        self.align = align

        # Store paths to clean and recorded files
        self.set_paths(tasks, ignore_paths)

        print(
            f"Initialized dataset with {len(self.file_paths)} files from {len(tasks)} tasks"
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> TrainChunk:
        paths = self.file_paths[idx]

        if self.return_chunks:
            clean_chunk, recorded_chunk = load_random_chunk_pair(paths=paths)

            return TrainChunk(clean_chunk, recorded_chunk, paths[1])
        else:
            clean_file, recorded_file = load_file_pair(paths=paths, align=self.align)

            return TrainChunk(clean_file, recorded_file, paths[1])

    def get_reference_batch(
        self, batch_size: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_elements = self.__len__()
        assert (
            batch_size < num_elements
        ), "Batch size must be smaller than number of elements in dataset"

        indices = np.random.choice(num_elements, batch_size)

        clean_chunks, recorded_chunks, _ = zip(
            *[self.__getitem__(idx) for idx in indices]
        )

        return (
            torch.stack(
                [torch.stack(clean_chunks), torch.stack(recorded_chunks)], dim=1
            )
            .squeeze()
            .to(device)
        )

    def set_paths(self, tasks: Union[str, List[str]], ignore_paths: List[str] = []):
        """
        Helper function to store paths for clean and recorded data for provided tasks list as well as ensuring they exist
        """
        # Get list of all file paths leading to clean and files
        file_paths_clean = self.get_file_paths(
            tasks=tasks, type=os.getenv("CLEAN_PATH"), ignore_paths=ignore_paths
        )
        file_paths_recorded = self.get_file_paths(
            tasks=tasks, type=os.getenv("RECORDED_PATH"), ignore_paths=ignore_paths
        )

        assert len(file_paths_clean) == len(
            file_paths_recorded
        ), "Number of clean and recorded files does not match!"

        # Combine lists into tuples of clean and recorded paths
        self.file_paths = list(zip(file_paths_clean, file_paths_recorded))

    def get_file_paths(
        self, tasks: Union[List[str], str], type: str, ignore_paths: List[str] = []
    ) -> List[str]:
        """
        Returns the file paths of the given type
        """

        tasks = [tasks] if isinstance(tasks, str) else tasks

        # For clean files in task 3 we need to get the corresponding files from task 2
        if type == os.getenv("CLEAN_PATH"):
            tasks = [
                "Task_2_Level_2"
                if task == "Task_3_Level_1"
                else "Task_2_Level_3"
                if task == "Task_3_Level_2"
                else task
                for task in tasks
            ]

        return [
            path
            for task_path in tasks
            for path in remove_file_paths(os.path.join(task_path, type), ignore_paths)
        ]
