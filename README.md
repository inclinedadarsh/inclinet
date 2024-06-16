# Inclinet - Simple neural network library

Inclinet is a simple neural network implementation in Python.
It uses pure python and numpy. Everything else has been implemented from scratch.

## Installation

Clone the repository and navigate to the project directory:

```sh
git clone https://github.com/inclinedadarsh/inclinet.git
cd inclinet
```

Install the package:

```sh
pip install .
```

### Alternate Installation
Alternatively, you can also run the following code block and it'll install the library without manually cloning it.

```sh
pip install git+https://github.com/yourusername/my_neural_network.git
```

## Requirements

The project requires NumPy 1.26.4:

```text
numpy==1.26.4
```

## Running Examples

To run the examples, navigate to the project directory and execute the example scripts:

```sh
python examples/example1.py
python examples/example2.py
```

## Basic Usage

### Creating and Training a Neural Network

Define and train a neural network:

```python
import numpy as np
from inclinet.train import train
from inclinet.neural_network import NeuralNet
from inclinet.layers import Linear, Relu
from inclinet.optimizer import SGD

# Example data
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

# Define the network
net = NeuralNet([
    Linear(input_size=2, output_size=5),
    Relu(),
    Linear(input_size=5, output_size=2)
])

# Train the network
train(net, inputs, targets, num_epoch=1000, optimizer=SGD(lr=0.01))
```

### Forward Pass and Prediction

Perform a forward pass through the network and make predictions:

```python
# Make predictions
for x in inputs:
    predicted = net.forward(x)
    print(f"Input: {x} -> Predicted: {predicted.round(2)}")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
