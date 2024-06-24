import csv
import logging
import pathlib
import re

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

REGEX_PATTERN = r"r(\d+)_c(\d+)"


class SubmissionDataset(Dataset):
    """
    Dataset holding the submission tuples.
    """

    def __init__(self, file_path: pathlib.Path, set_values_to_zero: bool = True) -> None:
        self.file_path = file_path
        self.data: list[list[tuple[int, int], float]] = []  # type: ignore

        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                id_str, rating_str = row
                match = re.match(REGEX_PATTERN, id_str)
                if match:
                    # both are 1-based
                    user_idx = int(match.group(1)) - 1
                    movie_idx = int(match.group(2)) - 1
                else:
                    raise ValueError(f"Id '{id_str}' does not match the expected pattern.")
                rating = float(rating_str.strip())
                self.data.append([(user_idx, movie_idx), 0.0 if set_values_to_zero else rating])

        logging.debug(f"Loaded a total of {len(self.data)} entries.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> list[tuple[int, int], float]:  # type: ignore
        return self.data[idx]
