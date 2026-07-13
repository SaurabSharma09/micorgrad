# Micrograd Complete Notes
### Andrej Karpathy — The Spelled Out Intro to Neural Networks and Backpropagation

---

## 1. What is Micrograd

Micrograd is a tiny autograd engine built from scratch. It implements backpropagation over a dynamically built computation graph. Everything in this library works at the scalar level — one number at a time — purely for learning purposes.

The same concepts power PyTorch, JAX, and every modern deep learning library.

---

## 2. The Value Class

The Value class is the foundation of everything. It wraps a single number and tracks gradients.

```python
class Value:
    def __init__(self, data, _children=()):
        self.data = data           # the actual number
        self.grad = 0.0            # gradient, starts at 0
        self._prev = set(_children) # parent nodes in graph
```

### Why each part exists

- `self.data` stores the number itself
- `self.grad` stores how much this value affects the final loss
- `self._prev` stores which Values created this one (the receipt)

### Dunder methods

```python
def __repr__(self):
    return f"Value(data={self.data})"   # cosmetic only, not needed for math

def __add__(self, other):
    out = Value(self.data + other.data, (self, other))
    return out

def __mul__(self, other):
    out = Value(self.data * other.data, (self, other))
    return out
```

`__repr__` is purely for readable printing. `__add__` and `__mul__` do the actual math AND record the computation graph by passing `(self, other)` as children.

### Key point

Without the Value class:
```python
a = 2.0
b = 3.0
c = a + b   # just a number, no history, no backprop possible
```

With the Value class:
```python
a = Value(2.0)
b = Value(3.0)
c = a + b   # Value(5.0) with c._prev = {a, b}
            # the graph is recorded, backprop is possible
```

---

## 3. What is a Derivative

A derivative answers one question:

> If I change this value a tiny bit, how much does the output change?

```python
h = 0.0001          # tiny nudge
d1 = f(x)           # output before
d2 = f(x + h)       # output after nudging x

derivative = (d2 - d1) / h
```

Example:
```
f(x) = x²

x = 3       → f(3) = 9
x = 3.0001  → f(3.0001) = 9.00060001

derivative = (9.00060001 - 9) / 0.0001 = 6.0
```

This tells you: if x increases by 1, f increases by 6.

---

## 4. The Computation Graph

Every operation between Value objects builds a graph:

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a * b    # d._prev = {a, b},  d.data = -6.0
e = d + c    # e._prev = {d, c},  e.data = 4.0
```

Graph:
```
a(2.0)   b(-3.0)
    \      /
     d(-6.0)    c(10.0)
          \      /
           e(4.0)
```

Each node knows its parents. Backpropagation walks this graph in reverse.

---

## 5. Chain Rule — The Heart of Backprop

The chain rule says:

```
gradient of input = upstream gradient × local derivative
```

In notation:
```
d(loss)/da = d(loss)/d(out) × d(out)/da
              upstream grad    local deriv
```

### Why chain rule

Because `a` does not directly touch loss. It goes through intermediate nodes:
```
a → out → loss

so: d(loss)/da = d(loss)/d(out) × d(out)/da
```

---

## 6. Backward Functions

Each operation stores a `_backward` function that pushes gradients to its inputs.

### Addition backward

```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other))

    def _backward():
        self.grad  += out.grad * 1   # local derivative of + is 1
        other.grad += out.grad * 1

    out._backward = _backward
    return out
```

Local derivative of addition is always 1 because:
```
out = a + b
if a increases by 1 → out increases by 1
rate = 1
```

### Multiplication backward

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other))

    def _backward():
        self.grad  += out.grad * other.data  # local derivative = other input
        other.grad += out.grad * self.data

    out._backward = _backward
    return out
```

Local derivative of multiplication:
```
out = a * b
d(out)/da = b   (other input)
d(out)/db = a   (other input)
```

### Tanh backward

```python
def tanh(self):
    t = math.tanh(self.data)
    out = Value(t, (self,))

    def _backward():
        self.grad += (1 - t**2) * out.grad   # derivative of tanh

    out._backward = _backward
    return out
```

Derivative of tanh:
```
o = tanh(n)
d(o)/d(n) = 1 - tanh(n)²  =  1 - o²
```

### Why += and not =

Because same variable can be used twice:
```python
b = a + a   # a appears twice

# WITHOUT +=
a.grad = 1.0   # first path
a.grad = 1.0   # second path overwrites → WRONG, total = 1.0

# WITH +=
a.grad += 1.0  # first path,  a.grad = 1.0
a.grad += 1.0  # second path, a.grad = 2.0 → CORRECT
```

---

## 7. Manual Backpropagation Example

Setup:
```python
x1 = Value(2.0),  w1 = Value(-3.0)
x2 = Value(0.0),  w2 = Value(1.0)
b  = Value(6.8813...)

n = x1*w1 + x2*w2 + b   # n.data = 0.8814
o = tanh(n)               # o.data = 0.7071
```

Backward pass step by step:

```
o.grad = 1.0                            (always starts here)

n.grad = o.grad × (1 - o.data²)
       = 1.0    × (1 - 0.7071²)
       = 1.0    × 0.5
       = 0.5

x1w1.grad = n.grad × 1 = 0.5           (addition passes through)
x2w2.grad = n.grad × 1 = 0.5
b.grad    = n.grad × 1 = 0.5

x1.grad = x1w1.grad × w1.data = 0.5 × (-3.0) = -1.5
w1.grad = x1w1.grad × x1.data = 0.5 × 2.0    =  1.0

x2.grad = x2w2.grad × w2.data = 0.5 × 1.0    =  0.5
w2.grad = x2w2.grad × x2.data = 0.5 × 0.0    =  0.0
```

---

## 8. Topological Sort and backward()

To automate backprop, we need to call `_backward` in the right order — output first, inputs last.

```python
def backward(self):
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```

Now you just call:
```python
o.backward()   # does everything automatically
```

---

## 9. Neuron Class

A single artificial neuron:

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        act = sum(wi*xi for wi, xi in zip(self.w, x)) + self.b
        out = act.tanh()
        return out
```

Example:
```python
n = Neuron(2)           # 2 inputs, random weights
x = [2.0, 3.0]

# internally:
# act = w1*2.0 + w2*3.0 + b
# out = tanh(act)
```

### Cell body vs activation function

```
inputs → [cell body: w*x + b] → [activation: tanh] → output

cell body   → raw weighted sum, any number (-∞ to +∞)
activation  → squashes into (-1, +1), adds non-linearity
```

### Why tanh

Without activation every layer collapses to one linear function no matter how deep. tanh introduces non-linearity allowing the network to learn complex patterns.

---

## 10. Layer Class

A group of neurons all receiving the same input:

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs
```

Example:
```python
L = Layer(3, 4)   # 3 inputs, 4 neurons
x = [2.0, 3.0, -1.0]

# all 4 neurons receive same x
# each produces different output (different weights)
# returns list of 4 Values
```

---

## 11. MLP Class

Multiple layers stacked together:

```python
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

Example:
```python
n = MLP(3, [4, 4, 1])

sz = [3] + [4, 4, 1] = [3, 4, 4, 1]

layers created:
Layer(3,4)   → 4 neurons × 3 weights = 12w + 4b = 16 params
Layer(4,4)   → 4 neurons × 4 weights = 16w + 4b = 20 params
Layer(4,1)   → 1 neuron  × 4 weights = 4w  + 1b = 5  params
total = 41 parameters
```

Forward pass:
```
x = [2.0, 3.0, -1.0]
    ↓
Layer(3,4) → [out1, out2, out3, out4]
    ↓
Layer(4,4) → [out1, out2, out3, out4]
    ↓
Layer(4,1) → [final_output]
```

---

## 12. parameters() Function

Collects all weights and biases into one flat list:

```python
class Neuron:
    def parameters(self):
        return self.w + [self.b]     # [w1, w2, w3, b]

class Layer:
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

Why needed:
```python
# WITHOUT parameters()
for layer in n.layers:
    for neuron in layer.neurons:
        for w in neuron.w:
            w.data -= 0.01 * w.grad  # 3 nested loops, messy

# WITH parameters()
for p in n.parameters():
    p.data -= 0.01 * p.grad          # 1 clean loop
```

---

## 13. Training Loop

Complete training loop:

```python
for iteration in range(100):

    # step 1: forward pass
    ypred = [n(x) for x in xs]

    # step 2: compute loss (mean squared error)
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # step 3: zero gradients BEFORE backward
    for p in n.parameters():
        p.grad = 0.0

    # step 4: backward pass
    loss.backward()

    # step 5: update weights
    for p in n.parameters():
        p.data -= 0.01 * p.grad

    print(iteration, loss.data)
```

---

## 14. Why zero_grad

Without zeroing gradients, `+=` accumulates across iterations:

```
iteration 1: w.grad += 0.5  → w.grad = 0.5  ← correct
iteration 2: w.grad += 0.5  → w.grad = 1.0  ← WRONG, doubled
iteration 3: w.grad += 0.5  → w.grad = 1.5  ← keeps growing
```

With zeroing:
```
iteration 1: w.grad = 0.0, then += 0.5 → w.grad = 0.5  ← correct
iteration 2: w.grad = 0.0, then += 0.5 → w.grad = 0.5  ← correct
iteration 3: w.grad = 0.0, then += 0.5 → w.grad = 0.5  ← correct
```

---

## 15. PyTorch Comparison

Micrograd and PyTorch do identical math:

| Micrograd | PyTorch |
|---|---|
| `Value(2.0)` | `torch.Tensor([2.0]).double()` |
| `self._prev` | `requires_grad=True` |
| `o._backward()` | `o.backward()` |
| `a.grad` | `a.grad` |
| scalar only | tensors (millions of numbers) |

The only real difference is scale. PyTorch operates on tensors for speed. The computation graph, chain rule, and gradient accumulation are identical.

---

## 16. Key Concepts Summary

| Concept | What it is | Simple explanation |
|---|---|---|
| Value | wrapped number | number that remembers its history |
| data | the number | actual value |
| grad | the gradient | how much this affects loss |
| _prev | parent nodes | who created this value |
| _backward | local backprop | how to push gradient to parents |
| forward pass | compute output | run math left to right |
| backward pass | compute gradients | run chain rule right to left |
| loss | error measure | how wrong is the network |
| weight | learnable param | the knob to turn |
| gradient | direction to turn | tells you how to fix the knob |
| learning rate | step size | how much to turn the knob |
| zero_grad | reset gradients | erase whiteboard before each round |
| parameters() | all weights + biases | flat list of everything learnable |

---

## 17. Architecture Hierarchy

```
Value      → single number with gradient tracking
  ↑
Neuron     → one brain cell (weights + bias + tanh)
  ↑
Layer      → group of neurons (all receive same input)
  ↑
MLP        → multiple layers stacked
  ↑
Training   → forward + loss + zero_grad + backward + update
```

Each level builds completely on the level below. This exact architecture is the foundation of GPT, just scaled to billions of parameters.

---

## 18. Important Rules to Remember

1. Derivative of addition = 1 always (gradient passes through unchanged)
2. Derivative of multiplication = the OTHER input's value
3. Derivative of tanh = 1 - tanh(x)²
4. o.grad always starts at 1.0 (output affects itself at rate 1)
5. Always use += for gradients (accumulation not overwrite)
6. Always zero_grad before backward() each iteration
7. nin = inputs per neuron = weights per neuron
8. nout = number of neurons = number of outputs from layer
9. parameters() flattens everything into one list
10. Every operation builds the graph, backward() walks it in reverse
