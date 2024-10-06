import os
from math import ceil
from random import sample
from typing import List, Tuple, Union

from dotenv import load_dotenv

from src.data.files import get_file_paths

load_dotenv()


PathPair = Tuple[str, str]


def sample_filepaths(
    tasks: List[str], sample_rate: float = 0.05, split_into: int = 1
) -> Union[List[PathPair], List[List[PathPair]]]:
    assert sample_rate > 0 and sample_rate <= 1, "Sample rate must be between 0 and 1!"
    assert len(tasks) > 0, "No tasks provided!"
    assert split_into > 0, "Split into must be greater than 0!"

    sampled_paths = []
    for task in tasks:
        task_path_clean = os.path.join(
            "Task_2_Level_2"
            if task == "Task_3_Level_1"
            else "Task_2_Level_3"
            if task == "Task_3_Level_2"
            else task,
            os.getenv("CLEAN_PATH"),
        )
        task_path_recorded = os.path.join(task, os.getenv("RECORDED_PATH"))

        clean_paths, recorded_paths = (
            get_file_paths(task_path_clean),
            get_file_paths(task_path_recorded),
        )

        assert len(clean_paths) == len(
            recorded_paths
        ), f"Number of clean and recorded files does not match in task {task}!"

        num_samples = ceil(len(clean_paths) * sample_rate)
        sampled_ids = sample(range(len(clean_paths)), num_samples)

        sampled_paths.extend(
            [
                (clean_path, recorded_path)
                for i, (clean_path, recorded_path) in enumerate(
                    zip(clean_paths, recorded_paths)
                )
                if i in sampled_ids
            ]
        )

    # Split into multiple alterning lists
    if split_into == 1:
        return sampled_paths

    split_sampled_paths = [sampled_paths[i::split_into] for i in range(split_into)]

    return split_sampled_paths


def remove_file_paths(path: str, ignore_paths: List[Tuple[str, str]]) -> List[str]:
    """
    Removes files from the list of files that should be ignored
    """
    file_paths = get_file_paths(path)

    if not ignore_paths:
        return file_paths

    clean_ignored_paths, recorded_ignored_paths = zip(*ignore_paths)

    return [
        file_path
        for file_path in file_paths
        if file_path not in clean_ignored_paths + recorded_ignored_paths
    ]
