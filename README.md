# Mini Autograd Engine

A lightweight scalar-based automatic differentiation engine built from scratch in Python, extended with a small neural-network module for constructing multi-layer perceptrons (MLPs).

The project is focused on understanding how automatic differentiation, computational graphs, neurons, layers, and neural networks work under the hood.

## Features

### Automatic Differentiation

- Computational graph construction with scalar `Value` objects
- Topological sorting for backpropagation
- Reverse-mode automatic differentiation
- Gradient accumulation
- Arithmetic operations: addition, subtraction, multiplication, division, and powers
- Non-linear operations: exponential and `tanh`

### Neural Network Module

The project also provides a minimal neural-network abstraction built on top of the `Value` class:

- `Neuron` — computes a weighted sum plus bias followed by `tanh`
- `Layer` — groups multiple neurons together
- `MLP` — stacks multiple layers to build a multi-layer perceptron
- Parameter collection through `parameters()`

## How the Neural Network Is Built

The neural-network module follows a simple hierarchy:

```text
Neuron
  ↓
Layer = multiple Neurons
  ↓
MLP = multiple Layers
```

For example:

```python
model = MLP(3, [4, 4, 1])
```

creates the architecture:

```text
3 inputs
   ↓
Layer 1: 4 neurons
   ↓
Layer 2: 4 neurons
   ↓
Layer 3: 1 neuron
   ↓
1 output
```

The output of one layer becomes the input to the next layer.

In other words:

```text
3 → 4 → 4 → 1
```

where each neuron maintains its own weights and bias using `Value` objects. Because those values participate in the computational graph, the neural-network operations can also be differentiated using the same autograd engine.

## Example

### Autograd

```python
from src.micrograd.engine import Value

a = Value(2.0)
b = Value(-3.0)

c = a * b
c.backward()

print(a.grad)
print(b.grad)
```

### Neural Network

The neural-network module can be constructed by specifying the number of inputs and the number of neurons in each layer:

```python
from src.micrograd.nn import Neuron, Layer, MLP

model = MLP(3, [4, 4, 1])
```

Then an input can be passed through the network:

```python
x = [Value(1.0), Value(2.0), Value(-1.0)]
out = model(x)
```

## Project Structure

```text
src/
├── micrograd/
│   ├── engine.py
│   └── computation_graph_visualizer.py
│
└── my_micrograd_practice/
    └── micrograd_from_scratch.ipynb
```

- `micrograd/engine.py` — scalar autograd engine and backpropagation
- `micrograd/computation_graph_visualizer.py` — computational graph visualization utilities
- `micrograd/nn.py` — neural network building module
- `my_micrograd_practice/` — practice and from-scratch implementation work

## Tech Stack

Python · OOP · Computational Graphs · Automatic Differentiation · Backpropagation · Neural Networks

## Status

Work in Progress — currently implementing the core autograd engine and a minimal neural-network module from scratch.
