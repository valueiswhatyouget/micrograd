"""
nn.py: a neural network built on top of our from-scratch engine.

A "neural network" sounds fancy but it's just: numbers (weights) multiplied
and added, passed through a squashing function, repeated in layers. Every one
of those operations is a `Value` from engine.py, so the same `.backward()`
that worked on a single multiply works on the whole network.
"""

import random
from engine import Value


class Module:
    """Base class: anything with parameters can reset their gradients."""

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    """One neuron: weighted sum of inputs + bias, then a nonlinearity."""

    def __init__(self, n_inputs, nonlin=True):
        # Scale the initial weights by 1/sqrt(fan-in) so the weighted sum
        # starts near unit variance instead of saturating tanh in its flat
        # tails (where gradients vanish and learning stalls).
        scale = n_inputs ** -0.5
        self.w = [Value(random.uniform(-1, 1) * scale) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        # sum(w_i * x_i) + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    """A row of neurons, all seeing the same inputs."""

    def __init__(self, n_inputs, n_outputs, nonlin=True):
        self.neurons = [Neuron(n_inputs, nonlin) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP(Module):
    """A multi-layer perceptron: layers stacked back to back."""

    def __init__(self, n_inputs, layer_sizes):
        sizes = [n_inputs] + layer_sizes
        # hidden layers use the nonlinearity; the final layer is linear
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=(i != len(layer_sizes) - 1))
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
