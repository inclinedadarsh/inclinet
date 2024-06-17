"""
XOR is an example of a function that can't be learnt
by a simple linear model.
"""

import numpy as np

from inclinet import train
from inclinet import NeuralNet
from inclinet import Linear, Sigmoid, Tanh, Relu

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
    Linear(input_size=2, output_size=5),
    # Tanh(),
    # Sigmoid(),
    Relu(),
    Linear(input_size=5, output_size=2)
])

train(net, inputs, targets)

for x, y, in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted.round(1), y)
