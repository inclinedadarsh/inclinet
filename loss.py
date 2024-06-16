"""
A loss function measures how good our prediction is.
We can use this to adjust parameters of our network.
"""

import numpy as np
from numpy import ndarray


class Loss:
    def loss(self, predicted: ndarray, target: ndarray) -> float:
        raise NotImplemented

    def grad(self, predicted: ndarray, target: ndarray) -> ndarray:
        raise NotImplemented


class MSE(Loss):
    """
    MSE is mean squared error.
    """

    def loss(self, predicted: ndarray, target: ndarray) -> float:
        return np.sum(np.power((predicted - target), 2))

    def grad(self, predicted: ndarray, target: ndarray) -> ndarray:
        return 2 * (predicted - target)