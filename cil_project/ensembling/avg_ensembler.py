import pathlib

import numpy as np
from cil_project.dataset import SubmissionDataset
from cil_project.utils import DATA_PATH

from .abstract_ensembler import AbstractEnsembler
from .utils import write_predictions_to_csv


class AvgEnsembler(AbstractEnsembler):

    def predict(self, submission_dataset: SubmissionDataset) -> pathlib.Path:
        """
        Generate predictions for the given input file and write them to the output folder.
        Expects the individual models to have stored the predictions in a file `<model_name>.csv`.

        :return: path to the generated output file.
        """

        # perform predictions for all predictors
        paths = [DATA_PATH / f"{name}.csv" for name in self.model_names]
        predictions = AbstractEnsembler.load_predictions(paths, submission_dataset.inputs)

        # no need for clipping, avg results should always be in [MIN_RATING, MAX_RATING]
        submission_dataset.predictions = np.expand_dims(np.mean(predictions, axis=1), axis=1)

        return write_predictions_to_csv(submission_dataset, self.__class__.__name__)
