# micrograd

> A tiny autograd engine + neural network library, built from scratch to understand what happens **under the hood**.

No PyTorch. No TensorFlow. Just Python, math, computational graphs, and backpropagation.

### ⚡ What's inside

* **Scalar Autograd** — builds computation graphs automatically
* **Backpropagation** — reverse-mode autodiff with topological sorting
* **Neural Networks** — `Neuron → Layer → MLP`
* **Operations** — `+`, `-`, `*`, `/`, `**`, `exp`, `tanh`
* **Gradients** — automatic gradient accumulation through the graph

### 🧠 Neural Network

```text
Input
  ↓
Neuron × N
  ↓
Layer
  ↓
Layer
  ↓
Output
```

Build an MLP by simply defining its architecture:

```python
model = MLP(3, [4, 4, 1])
```

which creates:

```text
3 → 4 → 4 → 1
```

Each weight and bias is a `Value`, meaning the entire network participates in the same computational graph and can be differentiated automatically.

### 🔬 Goal

I'm building this from scratch to understand the mechanics behind:

**Autograd → Backpropagation → Neural Networks**

before hiding them behind high-level frameworks.

### 🛠 Stack

`Python` · `OOP` · `Computational Graphs` · `Autodiff` · `Backpropagation` · `Neural Networks`


