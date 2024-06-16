from .rating_predictor import RatingPredictor  # noqa: F401


class DummyModel(RatingPredictor):
    def __init__(self) -> None:
        super().__init__("DummyModel")

    def predict(self, x: tuple[int, int]) -> float:
        return 5.0


class DummyModel2(RatingPredictor):
    def __init__(self) -> None:
        super().__init__("DummyModel2")

    def predict(self, x: tuple[int, int]) -> float:
        return 7.0
