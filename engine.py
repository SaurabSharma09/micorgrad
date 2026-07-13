"""
engine.py — the autograd engine

This is the heart of micrograd-explained.
A Value object wraps a single number and tracks:
    - its data (the actual number)
    - its gradient (how much it affects the final loss)
    - its parents (which Values created it)
    - its backward function (how to send gradients back)

Every operation between Value objects builds a computation graph.
When you call .backward(), gradients flow through that graph
using the chain rule at every single node.

This is exactly what PyTorch does internally when you call loss.backward().
The only difference is PyTorch works on tensors (millions of numbers at once).
We work on scalars (one number at a time) so the math is crystal clear.
"""

import math


class Value:
    """
    A single scalar value with automatic differentiation.

    Example:
        a = Value(2.0, label='a')
        b = Value(-3.0, label='b')
        c = a * b       # c.data = -6.0, c._prev = {a, b}
        c.backward()    # a.grad = -3.0, b.grad = 2.0
    """

    def __init__(self, data, _children=(), _op='', label=''):
        """
        Create a new Value.

        Args:
            data:      the actual number this Value holds
            _children: tuple of parent Values that created this one
                       empty () means this is a leaf node (input or weight)
            _op:       the operation that created this Value ('+', '*', 'tanh', etc.)
                       used only for visualization
            label:     human readable name (e.g. 'x1', 'w1', 'loss')
                       used only for visualization
        """
        self.data = data

        # gradient starts at 0.0
        # will be filled in during backward pass
        # meaning: how much does the final loss change if THIS value changes
        self.grad = 0.0

        # _prev stores the parent Values that created this one
        # we use a set so duplicates are handled automatically
        # e.g. if a + a: parents = {a} not {a, a}
        self._prev = set(_children)

        # _op records what operation created this Value
        # not used for math, only for the visualizer
        self._op = _op

        # label is a human-readable name
        # not used for math, only for the visualizer
        self.label = label

        # _backward is the function that computes gradients for the parents
        # default is a no-op (leaf nodes have no parents to send gradients to)
        # gets replaced by the actual backward function in each operation
        self._backward = lambda: None

    def __repr__(self):
        """
        Controls how the Value prints.
        Without this you would see: <Value object at 0x7f3c...>
        With this you see:          Value(data=2.0, grad=0.0)
        """
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # -------------------------------------------------------------------------
    # FORWARD OPERATIONS
    # Each operation:
    #   1. computes the output value (forward pass)
    #   2. defines _backward: how to send gradients back to parents
    #   3. records parents in _prev for graph traversal
    # -------------------------------------------------------------------------

    def __add__(self, other):
        """
        Addition: out = self + other

        Local derivatives:
            d(out)/d(self)  = 1   (if self increases by 1, out increases by 1)
            d(out)/d(other) = 1   (if other increases by 1, out increases by 1)

        Chain rule:
            self.grad  += out.grad * 1
            other.grad += out.grad * 1

        So gradient passes through addition completely unchanged.
        Addition is like a transparent pipe for gradients.
        """
        # handle plain numbers like a + 2.0
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # chain rule: upstream gradient * local derivative
            # local derivative of addition = 1 for both inputs
            # += because the same node might be used in multiple operations
            self.grad  += out.grad * 1
            other.grad += out.grad * 1

        out._backward = _backward
        return out

    def __mul__(self, other):
        """
        Multiplication: out = self * other

        Local derivatives:
            d(out)/d(self)  = other.data  (derivative of self*other w.r.t self is other)
            d(out)/d(other) = self.data   (derivative of self*other w.r.t other is self)

        Chain rule:
            self.grad  += out.grad * other.data
            other.grad += out.grad * self.data

        Each input gets the OTHER input's value as its gradient.
        This is why weights and inputs are connected during backprop.
        """
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        """
        Power: out = self ** exponent

        Only supports int or float exponents (not Value exponents).

        Local derivative:
            d(out)/d(self) = exponent * self.data ** (exponent - 1)
            (standard power rule from calculus)

        Example:
            out = x ** 2
            d(out)/dx = 2 * x
        """
        assert isinstance(exponent, (int, float)), "exponent must be int or float"

        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            # power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += out.grad * (exponent * self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def __neg__(self):
        """Negation: out = -self"""
        return self * -1

    def __sub__(self, other):
        """Subtraction: out = self - other = self + (-other)"""
        return self + (-other)

    def __radd__(self, other):
        """Right add: handles 2.0 + Value (Python calls this when left side fails)"""
        return self + other

    def __rsub__(self, other):
        """Right sub: handles 2.0 - Value"""
        return other + (-self)

    def __rmul__(self, other):
        """Right mul: handles 2.0 * Value"""
        return self * other

    def __truediv__(self, other):
        """Division: self / other = self * other^(-1)"""
        return self * other ** -1

    def __rtruediv__(self, other):
        """Right div: other / self"""
        return other * self ** -1

    # -------------------------------------------------------------------------
    # ACTIVATION FUNCTIONS
    # These introduce non-linearity into the network.
    # Without them, stacking layers is useless (collapses to one linear function).
    # -------------------------------------------------------------------------

    def tanh(self):
        """
        Hyperbolic tangent: out = tanh(self)

        Output range: (-1, +1)
        Used by Karpathy in the original micrograd video.

        Local derivative:
            d(tanh(x))/dx = 1 - tanh(x)^2 = 1 - out.data^2

        This is elegant: the gradient only needs the OUTPUT value,
        not the input. We already have out.data so it is cheap to compute.

        Example:
            tanh(0)    = 0.0
            tanh(1)    = 0.7616
            tanh(-1)   = -0.7616
            tanh(100)  ≈ 1.0
            tanh(-100) ≈ -1.0
        """
        x = self.data
        t = math.tanh(x)   # (e^x - e^-x) / (e^x + e^-x)

        out = Value(t, (self,), 'tanh')

        def _backward():
            # derivative of tanh: 1 - tanh(x)^2
            # we already computed t = tanh(x) in forward pass so reuse it
            self.grad += out.grad * (1 - t ** 2)

        out._backward = _backward
        return out

    def relu(self):
        """
        Rectified Linear Unit: out = max(0, self)

        Output range: [0, +inf)
        Most commonly used activation in modern networks.
        Simple but powerful. Most LLM feedforward layers use this.

        Local derivative:
            d(relu(x))/dx = 1 if x > 0
                          = 0 if x <= 0

        This is called the 'dead neuron' problem:
        if a neuron's input is always negative, its gradient is always 0
        and it never learns. Leaky ReLU fixes this but we keep it simple here.

        Example:
            relu(2.0)  = 2.0
            relu(-2.0) = 0.0
            relu(0.0)  = 0.0
        """
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            # gradient flows through only if input was positive
            # if input was negative, neuron was dead, gradient = 0
            self.grad += out.grad * (1.0 if self.data > 0 else 0.0)

        out._backward = _backward
        return out

    def sigmoid(self):
        """
        Sigmoid: out = 1 / (1 + e^(-self))

        Output range: (0, 1)
        Used for binary classification output layers.
        Also called the logistic function.

        Local derivative:
            d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
                             = out.data * (1 - out.data)

        Like tanh, the gradient only needs the output value.
        Very elegant derivative that reuses the forward pass result.

        Example:
            sigmoid(0)   = 0.5
            sigmoid(2)   = 0.88
            sigmoid(-2)  = 0.12
            sigmoid(100) ≈ 1.0
        """
        s = 1 / (1 + math.exp(-self.data))

        out = Value(s, (self,), 'sigmoid')

        def _backward():
            # derivative: sigmoid(x) * (1 - sigmoid(x))
            self.grad += out.grad * s * (1 - s)

        out._backward = _backward
        return out

    def exp(self):
        """
        Exponential: out = e^self

        Local derivative:
            d(e^x)/dx = e^x

        The exponential function is its own derivative.
        Used internally by softmax and cross entropy loss.
        """
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            # d(e^x)/dx = e^x = out.data
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def log(self):
        """
        Natural logarithm: out = ln(self)

        Local derivative:
            d(ln(x))/dx = 1/x

        Used in cross entropy loss: -sum(y * log(p))
        Input must be positive (log of negative is undefined).
        """
        assert self.data > 0, f"log undefined for non-positive values, got {self.data}"

        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            # d(ln(x))/dx = 1/x
            self.grad += out.grad * (1 / self.data)

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # BACKWARD PASS
    # -------------------------------------------------------------------------

    def backward(self):
        """
        Run backpropagation from this node back through the entire graph.

        This computes the gradient of this Value with respect to
        every leaf node (inputs and weights) in the computation graph.

        Algorithm:
            1. Set this node's gradient to 1.0 (it affects itself perfectly)
            2. Build topological order of all nodes in graph
            3. Call _backward() on each node in reverse topological order

        Topological order ensures every node's gradient is fully computed
        before we use it to compute gradients of its parents.

        Example:
            a = Value(2.0)
            b = Value(-3.0)
            c = a * b           # c._prev = {a, b}
            c.backward()
            # now a.grad = -3.0, b.grad = 2.0
        """
        # gradient of output with respect to itself is always 1
        # this is the seed that starts the entire backward pass
        self.grad = 1.0

        # build topological ordering of the entire computation graph
        # topological order: if A depends on B, B comes before A
        # we need this so gradients are always computed downstream first
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build_topo(parent)
                topo.append(node)

        build_topo(self)

        # walk graph in reverse topological order
        # each node's _backward() pushes gradients to its parents
        for node in reversed(topo):
            node._backward()

    # -------------------------------------------------------------------------
    # UTILITY
    # -------------------------------------------------------------------------

    def zero_grad(self):
        """
        Reset gradient to 0.0.

        Must be called before each training iteration.
        Without this, gradients accumulate across iterations because
        _backward uses += not =.

        Example without zero_grad:
            iteration 1: w.grad = 0.5
            iteration 2: w.grad = 1.0  <- wrong, accumulated
            iteration 3: w.grad = 1.5  <- keeps growing

        Example with zero_grad:
            iteration 1: w.grad = 0.5
            zero_grad()
            iteration 2: w.grad = 0.5  <- correct, fresh
        """
        self.grad = 0.0
