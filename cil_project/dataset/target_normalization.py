from enum import Enum, auto


class TargetNormalization(Enum):
    """
    Represents all possible target normalization strategies.
    """

    BY_USER = auto()
    BY_MOVIE = auto()
    BY_TARGET = auto()
    TO_TANH_RANGE = auto()  # targets are in [-1, 1] -> useful for tanh activation
