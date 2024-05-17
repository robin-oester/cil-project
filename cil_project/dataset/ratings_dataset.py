import csv
import logging
import pathlib
import re

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

REGEX_PATTERN = r"r(\d+)_c(\d+)"


class RatingsDataset(Dataset):
    """
    Dataset holding the ratings.
    """

    def __init__(self, file_path: pathlib.Path) -> None:
        self.file_path = file_path
        self.data: list[tuple[tuple[int, int], float]] = []

        with open(file_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                id_str, rating_str = row
                match = re.match(REGEX_PATTERN, id_str)
                if match:
                    row_index = int(match.group(1))
                    col_index = int(match.group(2))
                else:
                    raise ValueError(f"Id '{id_str}' does not match the expected pattern.")
                rating = float(rating_str.strip())
                self.data.append(((row_index, col_index), rating))

        logging.debug(f"Loaded a total of {len(self.data)} entries.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[tuple[int, int], float]:
        return self.data[idx]
