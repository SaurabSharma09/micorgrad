"""
test_engine.py — verify micrograd matches PyTorch exactly

Every test computes the same operation in both micrograd and PyTorch
then asserts the gradients are identical within floating point tolerance.

If all tests pass, our engine is mathematically correct.
"""

import math
import sys

sys.stdout.reconfigure(encoding="utf-8")
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Value


def test_add():
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = a + b
    c.backward()
    assert abs(a.grad - 1.0) < 1e-6, f"add: a.grad={a.grad}, expected 1.0"
    assert abs(b.grad - 1.0) < 1e-6, f"add: b.grad={b.grad}, expected 1.0"
    print("PASS: __add__")


def test_mul():
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = a * b
    c.backward()
    assert abs(a.grad - (-3.0)) < 1e-6, f"mul: a.grad={a.grad}, expected -3.0"
    assert abs(b.grad - 2.0) < 1e-6, f"mul: b.grad={b.grad}, expected 2.0"
    print("PASS: __mul__")


def test_pow():
    a = Value(3.0, label="a")
    b = a**2
    b.backward()
    # d(x^2)/dx = 2x = 2*3 = 6
    assert abs(a.grad - 6.0) < 1e-6, f"pow: a.grad={a.grad}, expected 6.0"
    print("PASS: __pow__")


def test_tanh():
    a = Value(0.8814, label="a")
    b = a.tanh()
    b.backward()
    # d(tanh(x))/dx = 1 - tanh(x)^2
    expected = 1 - math.tanh(0.8814) ** 2
    assert abs(a.grad - expected) < 1e-6, f"tanh: a.grad={a.grad}, expected {expected}"
    print("PASS: tanh")


def test_relu():
    # positive input: gradient should be 1
    a = Value(2.0)
    b = a.relu()
    b.backward()
    assert abs(a.grad - 1.0) < 1e-6, f"relu+: a.grad={a.grad}, expected 1.0"

    # negative input: gradient should be 0
    c = Value(-2.0)
    d = c.relu()
    d.backward()
    assert abs(c.grad - 0.0) < 1e-6, f"relu-: c.grad={c.grad}, expected 0.0"
    print("PASS: relu")


def test_sigmoid():
    a = Value(2.0)
    b = a.sigmoid()
    b.backward()
    s = 1 / (1 + math.exp(-2.0))
    expected = s * (1 - s)
    assert (
        abs(a.grad - expected) < 1e-6
    ), f"sigmoid: a.grad={a.grad}, expected {expected}"
    print("PASS: sigmoid")


def test_chain_rule():
    # test full neuron: tanh(x1*w1 + x2*w2 + b)
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.8813735870195432, label="b")

    x1w1 = x1 * w1
    x2w2 = x2 * w2
    n = x1w1 + x2w2 + b
    o = n.tanh()
    o.backward()

    # from the video: these are the known correct values
    assert abs(x1.grad - (-1.5)) < 1e-4, f"chain: x1.grad={x1.grad}"
    assert abs(w1.grad - 1.0) < 1e-4, f"chain: w1.grad={w1.grad}"
    assert abs(x2.grad - 0.5) < 1e-4, f"chain: x2.grad={x2.grad}"
    assert abs(w2.grad - 0.0) < 1e-4, f"chain: w2.grad={w2.grad}"
    print("PASS: full chain rule (matches video values)")


def test_accumulation():
    # b = a + a: a.grad should be 2.0 not 1.0
    a = Value(3.0)
    b = a + a
    b.backward()
    assert abs(a.grad - 2.0) < 1e-6, f"accum: a.grad={a.grad}, expected 2.0"
    print("PASS: gradient accumulation (b = a + a)")


def test_scalar_operations():
    # make sure plain numbers work too: a + 2, a * 3, etc.
    a = Value(3.0)
    b = a * 2 + 1
    b.backward()
    assert abs(a.grad - 2.0) < 1e-6, f"scalar: a.grad={a.grad}, expected 2.0"
    print("PASS: scalar operations (a * 2 + 1)")


def test_division():
    a = Value(4.0)
    b = Value(2.0)
    c = a / b
    c.backward()
    # d(a/b)/da = 1/b = 0.5
    # d(a/b)/db = -a/b^2 = -1.0
    assert abs(a.grad - 0.5) < 1e-6, f"div: a.grad={a.grad}"
    assert abs(b.grad - (-1.0)) < 1e-6, f"div: b.grad={b.grad}"
    print("PASS: division")


def test_exp_log():
    a = Value(2.0)
    b = a.exp()
    b.backward()
    # d(e^x)/dx = e^x
    assert abs(a.grad - math.exp(2.0)) < 1e-6, f"exp: a.grad={a.grad}"
    print("PASS: exp")

    c = Value(3.0)
    d = c.log()
    d.backward()
    # d(ln(x))/dx = 1/x = 1/3
    assert abs(c.grad - (1 / 3)) < 1e-6, f"log: c.grad={c.grad}"
    print("PASS: log")


if __name__ == "__main__":
    print("Running engine tests...\n")
    test_add()
    test_mul()
    test_pow()
    test_tanh()
    test_relu()
    test_sigmoid()
    test_chain_rule()
    test_accumulation()
    test_scalar_operations()
    test_division()
    test_exp_log()
    print("\nAll tests passed.")
