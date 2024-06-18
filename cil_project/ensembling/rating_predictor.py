import csv
import pathlib
import time
from abc import ABC, abstractmethod

from dataset.submission_dataset import SubmissionDataset


class RatingPredictor(ABC):
    def __init__(self, name: str) -> None:
        self.name: str = name

    @abstractmethod
    def predict(self, x: tuple[int, int]) -> float:
        raise NotImplementedError()

    def generate_predictions(self, input_file_path: pathlib.Path, output_folder_path: pathlib.Path) -> None:
        """
        Generate predictions for the given input file and write them to the output folder.
        """
        submission_dataset = SubmissionDataset(input_file_path)

        # perform predictions
        for idx, ((user_idx, movie_idx), _) in enumerate(submission_dataset.data):
            prediction = self.predict((user_idx, movie_idx))
            submission_dataset.data[idx][1] += prediction

        self.write_predictions_to_csv(submission_dataset.data, output_folder_path)

    def write_predictions_to_csv(
        self, predictions: list[list[tuple[int, int], float]], output_folder_path: pathlib.Path  # type: ignore
    ) -> None:
        """
        Write the predictions to a csv file.
        """
        # create csv file with predictions
        millis = int(time.time())
        output_file_path = output_folder_path / f"{self.name}_{millis}.csv"
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["Id", "Prediction"])
            for (user_idx, movie_idx), prediction in predictions:
                writer.writerow([f"r{user_idx+1}_c{movie_idx+1}", prediction])
