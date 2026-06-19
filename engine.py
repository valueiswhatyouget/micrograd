"""
engine.py: a tiny automatic-differentiation engine, written from scratch.

This is the whole magic behind every neural network, including the giant ones.
A `Value` wraps a single number and remembers how it was produced, so that
when you call `.backward()` we can walk the computation backwards and figure
out how much every input contributed to the final output. That contribution
is the *gradient*, and "backpropagation" is just the chain rule applied
mechanically over the graph.

No numpy, no PyTorch, just pure Python. About 120 lines. If you read it top to bottom
you will actually understand what "training a neural net" means.
"""

import math


class Value:
    """A scalar value and its gradient in a computation graph."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data          # the actual number
        self.grad = 0.0           # d(final output) / d(this value), filled in by backward()
        self._backward = lambda: None  # how to push gradient to our inputs
        self._prev = set(_children)    # the Values that produced this one
        self._op = _op            # what operation made this (for debugging/printing)

    # ---- the operations we support -------------------------------------
    # For each op we (1) compute the forward result and (2) define how to
    # send gradient backward to the inputs. That backward rule is just the
    # local derivative of the op, multiplied by the incoming gradient
    # (the chain rule).

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1  -> just pass the gradient through
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("only int/float powers are supported")
        out = Value(self.data ** power, (self,), f"**{power}")

        def _backward():
            # d(a**n)/da = n * a**(n-1)
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        # a smooth squashing nonlinearity, output in (-1, 1)
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            # d(tanh(a))/da = 1 - tanh(a)**2
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), "relu")

        def _backward():
            # gradient flows only where the input was positive
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    # ---- conveniences so we can write normal-looking math --------------
    def __neg__(self):          return self * -1
    def __sub__(self, other):   return self + (-other)
    def __radd__(self, other):  return self + other
    def __rmul__(self, other):  return self * other
    def __rsub__(self, other):  return other + (-self)
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ---- the heart of it: backpropagation ------------------------------
    def backward(self):
        """Fill in .grad for every Value that fed into this one."""
        # We must process nodes in reverse order of how they were built,
        # so a node's gradient is finalized before we push it to its inputs.
        # That ordering is a topological sort of the graph.
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        # The final output's gradient w.r.t. itself is 1.
        self.grad = 1.0
        # Walk backwards, letting each node hand gradient to its inputs.
        for node in reversed(topo):
            node._backward()
