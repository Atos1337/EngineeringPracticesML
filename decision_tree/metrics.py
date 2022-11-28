from collections import defaultdict
from typing import Callable

import numpy as np


def gini(x_array: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    mapp = defaultdict(int)
    for y_point in x_array:
        mapp[y_point] += 1
    mapp = np.array(list(mapp.values())) / x_array.shape[0]
    return mapp.dot(1 - mapp)


def entropy(x_array: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    mapp = defaultdict(int)
    for y_point in x_array:
        mapp[y_point] += 1
    mapp = np.array(list(mapp.values())) / x_array.shape[0]
    return -np.sum(mapp * np.log2(mapp))


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    return (
        criterion(np.concatenate([left_y, right_y])) * (len(left_y) + len(right_y))
        - len(right_y) * criterion(right_y)
        - len(left_y) * criterion(left_y)
    )
