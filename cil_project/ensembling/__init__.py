import os

from .avg_ensembler import AvgEnsembler  # noqa: F401
from .dummy_model import DummyModel, DummyModel2  # noqa: F401
from .meta_regressor import MetaRegressor  # noqa: F401
from .rating_predictor import RatingPredictor  # noqa: F401

files = os.listdir(os.path.dirname(__file__))
files.remove("__init__.py")
__all__ = [f[:-3] for f in files if f.endswith(".py")]
