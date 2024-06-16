import csv
import pathlib
from abc import ABC, abstractmethod
from datetime import datetime

from dataset.ratings_dataset import RatingsDataset


class RatingPredictor(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name

    @abstractmethod
    def predict(self, x: tuple[int, int]) -> float:
        raise NotImplementedError()

    def generate_predictions(self, input_file_path: pathlib.Path, output_folder_path: pathlib.Path) -> None:
        submission_dataset = RatingsDataset(input_file_path)
        predictions: dict[tuple[int, int], float] = {}

        for (user_idx, movie_idx), _ in submission_dataset:
            prediction = self.predict((user_idx, movie_idx))
            predictions[(user_idx, movie_idx)] = prediction

        # create csv file with predictions
        current_timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        output_file_path = output_folder_path / f"{self.model_name}_predictions_{current_timestamp}.csv"
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["Id", "Prediction"])
            for (user_idx, movie_idx), prediction in predictions.items():
                writer.writerow([f"r{user_idx+1}_c{movie_idx+1}", prediction])
