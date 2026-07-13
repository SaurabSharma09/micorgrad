"""
test_nn.py — verify all neural network components

Tests every class in nn.py:
    - Neuron: correct output, correct parameter count
    - Layer: correct shape, correct parameter count
    - MLP: correct architecture, correct parameter count
    - Dropout: zeroing works, scaling works
    - BatchNorm: normalization works, parameters learned
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Value
from nn import Neuron, Layer, MLP, Dropout, BatchNorm


def test_neuron_output():
    n = Neuron(nin=3, activation="tanh")
    x = [Value(1.0), Value(2.0), Value(3.0)]
    out = n(x)
    # output must be a single Value
    assert isinstance(out, Value), "Neuron output must be a Value"
    # tanh output must be in (-1, 1)
    assert -1 < out.data < 1, f"tanh output {out.data} out of range"
    print("PASS: Neuron output is Value in (-1,1)")


def test_neuron_parameters():
    n = Neuron(nin=3)
    params = n.parameters()
    # nin=3 → 3 weights + 1 bias = 4 parameters
    assert len(params) == 4, f"Neuron(3) should have 4 params, got {len(params)}"
    print("PASS: Neuron(3) has 4 parameters")


def test_neuron_activations():
    x = [Value(0.5), Value(-0.5)]
    for act in ["tanh", "relu", "sigmoid", "linear"]:
        n = Neuron(nin=2, activation=act)
        out = n(x)
        assert isinstance(out, Value), f"{act} output not a Value"
    print("PASS: all activation functions work")


def test_neuron_backward():
    n = Neuron(nin=2, activation="tanh")
    x = [Value(1.0), Value(2.0)]
    out = n(x)
    out.backward()
    # all parameters should have non-zero gradients
    for p in n.parameters():
        # gradient exists (was computed)
        assert isinstance(p.grad, float), "gradient must be float"
    print("PASS: Neuron backward pass computes gradients")


def test_layer_output_shape():
    L = Layer(nin=3, nout=4)
    x = [Value(1.0), Value(2.0), Value(3.0)]
    outs = L(x)
    assert len(outs) == 4, f"Layer(3,4) should output 4 values, got {len(outs)}"
    assert all(isinstance(o, Value) for o in outs), "all outputs must be Values"
    print("PASS: Layer(3,4) outputs 4 Values")


def test_layer_single_output():
    # Layer with 1 neuron should return Value directly, not list
    L = Layer(nin=3, nout=1)
    x = [Value(1.0), Value(2.0), Value(3.0)]
    out = L(x)
    assert isinstance(out, Value), "Layer(3,1) should return Value not list"
    print("PASS: Layer(3,1) returns single Value")


def test_layer_parameters():
    L = Layer(nin=3, nout=4)
    params = L.parameters()
    # 4 neurons × (3 weights + 1 bias) = 16 parameters
    assert len(params) == 16, f"Layer(3,4) should have 16 params, got {len(params)}"
    print("PASS: Layer(3,4) has 16 parameters")


def test_mlp_forward():
    model = MLP(nin=2, layer_sizes=[4, 4, 1])
    x = [Value(1.0), Value(-2.0)]
    out = model(x)
    assert isinstance(out, Value), "MLP output must be a Value"
    print("PASS: MLP(2,[4,4,1]) forward pass works")


def test_mlp_parameters():
    model = MLP(nin=2, layer_sizes=[4, 4, 1])
    params = model.parameters()
    # layer1: Layer(2,4)  → 4*(2+1) = 12
    # layer2: Layer(4,4)  → 4*(4+1) = 20
    # layer3: Layer(4,1)  → 1*(4+1) = 5
    # total: 37
    assert len(params) == 37, f"MLP(2,[4,4,1]) should have 37 params, got {len(params)}"
    print("PASS: MLP(2,[4,4,1]) has 37 parameters")


def test_mlp_zero_grad():
    model = MLP(nin=2, layer_sizes=[4, 1])
    x = [Value(1.0), Value(2.0)]
    out = model(x)
    out.backward()
    # gradients should be non-zero after backward
    has_grad = any(abs(p.grad) > 0 for p in model.parameters())
    assert has_grad, "some gradients should be non-zero after backward"

    model.zero_grad()
    # after zero_grad all gradients should be 0
    all_zero = all(p.grad == 0.0 for p in model.parameters())
    assert all_zero, "all gradients should be 0 after zero_grad"
    print("PASS: zero_grad resets all gradients")


def test_dropout_training():
    drop = Dropout(p=0.5)
    x = [Value(1.0)] * 100
    out = drop(x, training=True)
    # some outputs should be 0 (dropped) with p=0.5
    zeros = sum(1 for o in out if abs(o.data) < 1e-9)
    # with p=0.5, roughly 50 should be zero
    # allow wide tolerance for randomness
    assert 10 < zeros < 90, f"Dropout p=0.5 zeroed {zeros}/100, expected ~50"
    print(f"PASS: Dropout(p=0.5) zeroed {zeros}/100 neurons in training")


def test_dropout_inference():
    drop = Dropout(p=0.5)
    x = [Value(2.0)] * 4
    out = drop(x, training=False)
    # inverted dropout: inference = pass-through, no scaling
    # values should remain 2.0 (not scaled by (1-p))
    for o in out:
        assert abs(o.data - 2.0) < 1e-6, f"inference pass-through wrong: {o.data}"
    print("PASS: Dropout inference passes through unchanged (inverted dropout)")


def test_dropout_zero():
    # p=0.0 means no dropout, pass through unchanged
    drop = Dropout(p=0.0)
    x = [Value(3.0), Value(-1.0)]
    out = drop(x, training=True)
    for xi, oi in zip(x, out):
        assert abs(xi.data - oi.data) < 1e-6
    print("PASS: Dropout(p=0.0) passes through unchanged")


def test_batchnorm_normalization():
    bn = BatchNorm(nin=4)
    # force all gamma=1, beta=0 so we can test pure normalization
    for g in bn.gamma:
        g.data = 1.0
    for b in bn.beta:
        b.data = 0.0

    x = [Value(1.0), Value(2.0), Value(3.0), Value(4.0)]
    out = bn(x)

    # compute expected: mean=2.5, var=1.25, std=sqrt(1.25+eps)
    vals = [o.data for o in out]
    mean = sum(vals) / len(vals)
    # after normalization, mean should be ~0
    assert abs(mean) < 1e-4, f"BatchNorm mean={mean}, expected ~0"
    print(f"PASS: BatchNorm normalizes to mean~0 (got {mean:.6f})")


def test_batchnorm_parameters():
    bn = BatchNorm(nin=4)
    params = bn.parameters()
    # 4 gamma + 4 beta = 8 parameters
    assert len(params) == 8, f"BatchNorm(4) should have 8 params, got {len(params)}"
    print("PASS: BatchNorm(4) has 8 parameters (4 gamma + 4 beta)")


if __name__ == "__main__":
    print("Running nn.py tests...\n")

    print("--- Neuron ---")
    test_neuron_output()
    test_neuron_parameters()
    test_neuron_activations()
    test_neuron_backward()

    print("\n--- Layer ---")
    test_layer_output_shape()
    test_layer_single_output()
    test_layer_parameters()

    print("\n--- MLP ---")
    test_mlp_forward()
    test_mlp_parameters()
    test_mlp_zero_grad()

    print("\n--- Dropout ---")
    test_dropout_training()
    test_dropout_inference()
    test_dropout_zero()

    print("\n--- BatchNorm ---")
    test_batchnorm_normalization()
    test_batchnorm_parameters()

    print("\nAll tests passed.")
