"""
We use an optimizer to adjust parameters of our network
based on the gradients computed during backpropagation
"""

from neural_network import NeuralNet


class Optimizer:
    def step(self, ned: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad