import csv
import pathlib
import time

from cil_project.dataset import SubmissionDataset
from cil_project.utils import DATA_PATH


def write_predictions_to_csv(submission_dataset: SubmissionDataset, predictor_name: str) -> pathlib.Path:
    """
    Write the predictions to a csv file. The predictions are aggregated in the submission dataset.

    :param submission_dataset: the submission dataset containing the predictions.
    :param predictor_name: name of the predictor.
    :return: path to the generated file.
    """

    # create csv file with predictions
    millis = int(time.time())
    output_file_path = DATA_PATH / f"{predictor_name}_{millis}.csv"
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["Id", "Prediction"])
        for (user_idx, movie_idx), prediction in zip(
            submission_dataset.inputs, submission_dataset.predictions.squeeze()
        ):
            writer.writerow([f"r{user_idx + 1}_c{movie_idx + 1}", prediction])

    return output_file_path
