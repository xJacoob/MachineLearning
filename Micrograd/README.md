# Micrograd — Building a Tiny Autograd Engine from Scratch

A from-scratch implementation of a scalar-valued autograd engine, following Andrej Karpathy's ["The spelled-out intro to neural networks and backpropagation: building micrograd"](https://www.youtube.com/watch?v=VMj-3S1tku0) (Neural Networks: Zero to Hero, Lecture 1).

## What this is

Every modern deep learning framework (PyTorch, JAX) relies on **automatic differentiation** — the ability to compute the gradient of a loss function with respect to every parameter in a model, no matter how deep or complex, without ever deriving that gradient by hand. This project builds that mechanism from first principles, on plain scalars, to understand exactly what `.backward()` is doing under the hood before treating it as a black box.

## What's implemented

**`Value` class** — wraps a single scalar and tracks:
- `data` — the actual number
- `grad` — the derivative of the final output with respect to this value (accumulates via `+=`, not `=`, to correctly handle variables used more than once in an expression)
- `_prev` / `_op` — pointers to the values and operation that produced this one, forming a computational graph
- `_backward` — a closure implementing that operation's local chain-rule step

Supported operations, each with its own forward pass and local gradient rule: `+`, `-`, `*`, `/`, `**` (constant powers), unary negation, `exp()`, `tanh()`.

**`backward()`** — builds a topological ordering of the graph starting from the output node, then calls each node's `_backward()` in reverse order, propagating gradients backward through the entire expression via repeated application of the chain rule.

**`Neuron` / `Layer` / `MLP`** — a tiny neural net library on top of the engine, mirroring PyTorch's API:
- `Neuron`: `tanh(w · x + b)` for a random weight vector and bias
- `Layer`: a list of neurons evaluated on the same input (one row of a weight matrix each)
- `MLP`: a sequence of layers applied one after another

**Training loop** — a 3 → 4 → 4 → 1 MLP trained with plain gradient descent (no optimizer library) on a 4-example toy binary classification dataset, using mean squared error as the loss. Loss dropped from 6.50 to 0.0046 over 50 iterations; final predictions matched the targets almost exactly.

**Verification against PyTorch** — rebuilt the same expression using `torch.tensor(..., requires_grad=True)` and confirmed the gradients from `.backward()` match the hand-rolled engine exactly.

## Key learnings

- **Local derivative vs. global gradient**: a node only ever knows its own local rule (e.g. for `a*b`, the local derivative w.r.t. `a` is just `b`). The chain rule is what turns a chain of these local rules into the derivative of the whole expression — each `_backward()` call multiplies its local derivative by the gradient it already received from downstream.
- **Why topological sort is required**: a node's incoming gradient must be fully accumulated (from every path that flows through it) before its own `_backward()` runs — calling it too early uses an incomplete gradient.
- **Why gradients accumulate (`+=`), not overwrite (`=`)**: any variable reused more than once in an expression receives a separate gradient contribution from each path it's part of; overwriting silently drops all but the last one.
- **Neuron → Layer → MLP maps directly onto matrix multiplication**: one neuron is one row of a weight matrix; a layer is the full matrix; stacking layers is just applying several matrices in sequence.
- **Why this approach scales where a hand-derived gradient formula doesn't**: a manually derived gradient is tied to one specific architecture and grows unmanageable as the model grows. An autograd engine only needs a small, fixed set of local rules (`+`, `*`, `tanh`, ...) — arbitrarily deep and complex graphs are built by composing those same simple rules, never requiring one giant formula to be derived by hand.