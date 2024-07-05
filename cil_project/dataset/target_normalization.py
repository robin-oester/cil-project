from enum import Enum


class TargetNormalization(Enum):
    """
    Represents all possible target normalization strategies.

    Warning! Do not compare using the `==` operator directly. Use .value and then check for equality.
    """

    BY_USER = 1
    BY_MOVIE = 2
    BY_TARGET = 3
    TO_TANH_RANGE = 4  # targets are in [-1, 1] -> useful for tanh activation
