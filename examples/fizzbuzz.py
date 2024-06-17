"""
For each of the number between 1 and 100:
* if the number is divisible by 3, print fizz
* if the number is divisible by 5, print buzz
* if the number is divisible by both, 3 and 5, print fizzbuzz
* otherwise just print the number itself
"""

from typing import List

import numpy as np

from inclinet import train
from inclinet import NeuralNet
from inclinet import Linear, Tanh
from inclinet import SGD


def fizz_buzz_encode(x: int) -> List[int]:
    """
    Encodes the given input 'x' according to fizz buzz category

    :param x: Integer to encode
    :return: [number, fizz, buzz, fizzbuzz]
    """
    if x % 15 == 0:
        return [0, 0, 0, 1]
    elif x % 5 == 0:
        return [0, 0, 1, 0]
    elif x % 3 == 0:
        return [0, 1, 0, 0]
    else:
        return [1, 0, 0, 0]


def binary_encode(x: int) -> List[int]:
    """
    10 digit binary encoding of input 'x'
    :param x: Input integer
    :return: Binary output
    """
    return [x >> i & 1 for i in range(10)]


# We'll train the neural network on number 101 to 1024
# and test it on number 1 to 100
# to make sure we aren't testing on the training set

X_train = np.array([
    binary_encode(x) for x in range(101, 1024)
])

y_train = np.array([
    fizz_buzz_encode(x) for x in range(101, 1024)
])


net = NeuralNet([
    Linear(input_size=10, output_size=50),
    Tanh(),
    Linear(input_size=50, output_size=4)
])

train(
    net,
    X_train,
    y_train,
    num_epoch=2000,
    optimizer=SGD(lr=0.001)
)

for x in range(1, 101):
    pred = net.forward(binary_encode(x))
    pred_idx = np.argmax(pred)
    y_test_idx = np.argmax(fizz_buzz_encode(x))
    labels = [str(x), 'fizz', 'buzz', 'fizzbuzz']
    print(x, labels[pred_idx], labels[y_test_idx])

# Following code is to print the accuracy
y_test = np.array([
    fizz_buzz_encode(x) for x in range(1, 101)
])

X_test = np.array([
    binary_encode(x) for x in range(1, 101)
])

y_pred = net.forward(X_test)

y_test_idx = np.argmax(y_test, axis=1)
y_pred_idx = np.argmax(y_pred, axis=1)

correct_predictions = np.sum(y_test_idx == y_pred_idx)

total_predictions = y_test_idx.shape[0]
accuracy = correct_predictions / total_predictions

print(f"Accuracy: {accuracy:.2f}")