import numpy as np

from .rating_predictor import RatingPredictor


class DummyModel(RatingPredictor):

    def get_name(self) -> str:
        return self.__class__.__name__

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return 5.0 * np.ones((inputs.shape[0], 1))


class DummyModel2(RatingPredictor):
    def get_name(self) -> str:
        return self.__class__.__name__

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return 3.0 * np.ones((inputs.shape[0], 1))
