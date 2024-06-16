"""
Here's a function that can train the neural network
"""

from numpy import ndarray
from nn import NeuralNet
from loss import Loss, MSE
from optimizer import Optimizer, SGD
from data import DataIterator, BatchIterator


def train(net: NeuralNet,
          inputs: ndarray,
          targets: ndarray,
          num_epoch: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)
