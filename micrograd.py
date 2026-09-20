"""A tiny scalar-valued autograd engine and neural network library.

This is a self-contained reimplementation of Andrej Karpathy's micrograd.
It is intentionally small: every Value stores one scalar and the local rule
needed to propagate a gradient to its parents.
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


class Value:
    """A scalar value that remembers the expression used to create it."""

    def __init__(
        self,
        data: float,
        _children: Iterable["Value"] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self.grad = 0.0

        # Internal graph information used during backpropagation.
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    @staticmethod
    def _coerce(other: float | "Value") -> "Value":
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other: float | "Value") -> "Value":
        other = self._coerce(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(self + other)/dself = 1 and d(...)/dother = 1.
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: float | "Value") -> "Value":
        return self + other

    def __mul__(self, other: float | "Value") -> "Value":
        other = self._coerce(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # Product rule: d(a*b)/da = b and d(a*b)/db = a.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: float | "Value") -> "Value":
        return self * other

    def __pow__(self, other: int | float) -> "Value":
        if not isinstance(other, (int, float)):
            raise TypeError("Value only supports int or float powers")
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: float | "Value") -> "Value":
        return self + (-self._coerce(other))

    def __rsub__(self, other: float | "Value") -> "Value":
        return self._coerce(other) + (-self)

    def __truediv__(self, other: float | "Value") -> "Value":
        return self * self._coerce(other) ** -1

    def __rtruediv__(self, other: float | "Value") -> "Value":
        return self._coerce(other) * self**-1

    def exp(self) -> "Value":
        result = math.exp(self.data)
        out = Value(result, (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        result = math.tanh(self.data)
        out = Value(result, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(max(0.0, self.data), (self,), "ReLU")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        """Apply reverse-mode autodiff to this value's entire graph.

        Parents must be processed after their children.  A topological sort
        gives exactly that order when traversed in reverse.
        """
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(value: Value) -> None:
            if value not in visited:
                visited.add(value)
                for child in value._prev:
                    build_topo(child)
                topo.append(value)

        build_topo(self)
        self.grad = 1.0
        for value in reversed(topo):
            value._backward()


class Module:
    """Base class for objects containing trainable Value parameters."""

    def parameters(self) -> list[Value]:
        return []

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad = 0.0


class Neuron(Module):
    def __init__(self, nin: int, nonlin: bool = True) -> None:
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: Sequence[float | Value]) -> Value:
        if len(x) != len(self.w):
            raise ValueError(f"expected {len(self.w)} inputs, got {len(x)}")
        activation = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return activation.relu() if self.nonlin else activation

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        kind = "ReLU" if self.nonlin else "Linear"
        return f"{kind}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin: int, nout: int, **kwargs: bool) -> None:
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: Sequence[float | Value]) -> Value | list[Value]:
        outputs = [neuron(x) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self) -> list[Value]:
        return [parameter for neuron in self.neurons for parameter in neuron.parameters()]

    def __repr__(self) -> str:
        return f"Layer([{', '.join(map(str, self.neurons))}])"


class MLP(Module):
    """A fully connected multilayer perceptron."""

    def __init__(self, nin: int, nouts: Sequence[int]) -> None:
        sizes = [nin, *nouts]
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x: Sequence[float | Value]) -> Value | list[Value]:
        result: Value | list[Value] | Sequence[float | Value] = x
        for layer in self.layers:
            # A non-final one-neuron layer returns a scalar; normalize it for
            # the next layer without complicating Layer's friendly API.
            inputs = [result] if isinstance(result, Value) else result
            result = layer(inputs)
        return result

    def parameters(self) -> list[Value]:
        return [parameter for layer in self.layers for parameter in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP([{', '.join(map(str, self.layers))}])"


def demo() -> None:
    """Train a tiny binary classifier, matching the micrograd lesson."""
    random.seed(42)
    model = MLP(3, [4, 4, 1])
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    for step in range(100):
        predictions = [model(x) for x in xs]
        loss = sum((prediction - target) ** 2 for prediction, target in zip(predictions, ys))

        model.zero_grad()
        loss.backward()

        learning_rate = 0.05
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

        if step % 10 == 0 or step == 99:
            print(f"step {step:3d} | loss {loss.data:.6f}")

    print("predictions:", [round(model(x).data, 3) for x in xs])


if __name__ == "__main__":
    demo()
