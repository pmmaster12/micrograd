# Mini Autograd Engine

A lightweight scalar-based automatic differentiation engine built from scratch in Python, inspired by the core concepts of [micrograd](https://github.com/karpathy/micrograd).

## Features

* Computational graph construction
* Topological sorting for backpropagation
* Reverse-mode automatic differentiation
* Gradient accumulation
* Support for arithmetic operations, powers, exponentials, and `tanh`

## Example

```python
a = Value(2.0)
b = Value(-3.0)

c = a * b
c.backward()

print(a.grad)
print(b.grad)
```

## Tech Stack

Python · OOP · Computational Graphs · Automatic Differentiation

## Status

Work in Progress — expanding toward a minimal neural-network framework.
