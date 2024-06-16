"""
XOR is an example of a function that can't be learnt
by a simple linear model.
"""

import numpy as np

from train import train
from nn import NeuralNet
from layers import Linear, Tanh

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_size=2, output_size=2),
    Tanh(),
    Linear(input_size=2, output_size=2)
])

train(net, inputs, targets)

for x, y, in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted.round(1), y)
