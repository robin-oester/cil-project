import csv
import logging
import pathlib
import re

import numpy as np
import pandas as pd
from torch.utils.data import Dataset  # pylint: disable=E0401

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
                    # both are 1-based
                    user_idx = int(match.group(1)) - 1
                    movie_idx = int(match.group(2)) - 1
                else:
                    raise ValueError(f"Id '{id_str}' does not match the expected pattern.")
                rating = float(rating_str.strip())
                self.data.append(((user_idx, movie_idx), rating))

        logging.debug(f"Loaded a total of {len(self.data)} entries.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[tuple[int, int], float]:
        return self.data[idx]

    def get_data_matrix(self, num_users: int = 10000, num_movies: int = 1000) -> np.ndarray:
        """
        Returns the dataset as a matrix. Each non-zero value marks an observed rating.
        """

        ratings = np.zeros((num_users, num_movies), dtype=float)

        for idx, ((user_id, movie_id), rating) in enumerate(self.data):
            if ratings[user_id][movie_id] > 0:
                raise ValueError(f"Duplicate rating at index '{idx}'.")
            ratings[user_id][movie_id] = rating
        return ratings

    def get_data_frame(self) -> pd.DataFrame:
        """
        Returns the dataset as a pandas DataFrame.
        """
        data = self.data
        df = pd.DataFrame(data, columns=["user_movie", "rating"])
        df[["user", "movie"]] = pd.DataFrame(df["user_movie"].tolist(), index=df.index)
        df = df.drop(columns=["user_movie"])
        return df
