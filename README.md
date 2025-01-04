# Gradease: A Simple Neural Network Library

**Gradease** is a lightweight Python library designed to facilitate experimentation with neural network architectures. This project implements core concepts of machine learning, including **gradient descent**, from scratch, providing an intuitive and flexible interface for building and training neural networks.

## Features

-   Implementation of **autograd** for automatic differentiation.
-   Basic building blocks for creating neural networks:
    -   **Neuron**: A single unit of computation.
    -   **Layer**: A collection of neurons.
    -   **MLP**: A multi-layer perceptron for deep learning.
-   Support for non-linear activation functions like **ReLU**.
-   Customizable and extensible design to explore various neural network architectures.

---

## Installation

To use Gradease, clone this repository and include the library in your Python project:

```bash
git clone https://github.com/your-repo/gradease.git
```

---

## Code Overview

### Core Components

1. **`Value` Class**

    - Represents a scalar value with its associated gradient for backpropagation.
    - Supports basic operations (`+`, `-`, `*`, `/`, `**`) with gradient computation.
    - Includes a `relu` method for activation and a `backward` method for gradient propagation.
    - Example:
        ```python
        a = Value(2.0)
        b = Value(3.0)
        c = a * b + a**2
        c.backward()
        print(a.grad)  # Gradient of `a` with respect to `c`
        ```

2. **`Module` Class**

    - Base class for all neural network components.
    - Provides utility methods like `parameters()` and `zero_grad()`.

3. **`Neuron` Class**

    - Implements a single neuron with optional non-linearity (ReLU).
    - Example:
        ```python
        n = Neuron(3)  # Neuron with 3 inputs
        output = n([Value(1.0), Value(2.0), Value(3.0)])
        print(output)
        ```

4. **`Layer` Class**

    - A collection of neurons forming a layer in the network.
    - Example:
        ```python
        l = Layer(3, 4)  # Layer with 3 inputs and 4 outputs
        output = l([Value(1.0), Value(2.0), Value(3.0)])
        print(output)
        ```

5. **`MLP` Class**
    - A multi-layer perceptron (feedforward neural network) built using layers.
    - Example:
        ```python
        mlp = MLP(3, [4, 4, 1])  # 3 inputs, two hidden layers with 4 neurons each, and 1 output
        output = mlp([Value(1.0), Value(2.0), Value(3.0)])
        print(output)
        ```

---

## Usage

### Building a Neural Network

```python
from gradease.engine import Value, Neuron, Layer, MLP

# Create an MLP with 3 inputs, two hidden layers, and 1 output
model = MLP(3, [4, 4, 1])

# Forward pass
inputs = [Value(1.0), Value(2.0), Value(3.0)]
output = model(inputs)

# Compute gradients
output.backward()

# Print model parameters and gradients
for p in model.parameters():
    print(p)
```

---

## Key Concepts

1. **Gradient Descent**

    - The `Value` class computes gradients automatically using reverse-mode differentiation.
    - Gradients are accumulated and can be reset using `zero_grad()`.

2. **Modular Design**

    - Build complex architectures by combining `Neuron`, `Layer`, and `MLP`.

3. **Non-Linearity**
    - Supports activation functions like ReLU for better learning capacity.

---

## Extensibility

You can extend the library by adding custom activation functions, layers, or optimizers. For example:

```python
class CustomNeuron(Neuron):
    def custom_activation(self, x):
        # Define your custom activation
        return x * (x > 0)
```

---
