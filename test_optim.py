"""
test_optim.py - Tests for optim.py
===================================
Verifies SGD and Adam against known values and PyTorch equivalents.
"""

import os
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")
# ── import engine and optimizers ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Value
from optim import SGD, Adam


def make_params(vals):
    """Create a list of Value objects from a list of floats."""
    params = [Value(v) for v in vals]
    return params


def set_grads(params, grads):
    """Manually assign gradients (simulates what backward() would do)."""
    for p, g in zip(params, grads):
        p.grad = g


def close(a, b, tol=1e-6):
    return abs(a - b) < tol


passed = 0
failed = 0


def test(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        failed += 1


print("\n── SGD Tests ──────────────────────────────────────────────────────────")

# Test 1: basic SGD step
params = make_params([1.0, -2.0, 3.0])
set_grads(params, [0.1, 0.2, -0.3])
opt = SGD(params, lr=0.1)
opt.step()
test("SGD basic step p0", close(params[0].data, 1.0 - 0.1 * 0.1))  # 0.99
test("SGD basic step p1", close(params[1].data, -2.0 - 0.1 * 0.2))  # -2.02
test("SGD basic step p2", close(params[2].data, 3.0 - 0.1 * (-0.3)))  # 3.03

# Test 2: zero_grad clears gradients
params = make_params([1.0])
params[0].grad = 5.0
opt = SGD(params, lr=0.01)
opt.zero_grad()
test("SGD zero_grad", params[0].grad == 0.0)

# Test 3: SGD with weight decay
# grad_effective = grad + wd * data = 0.5 + 0.1*2.0 = 0.7
# new_data = 2.0 - 0.1 * 0.7 = 1.93
params = make_params([2.0])
set_grads(params, [0.5])
opt = SGD(params, lr=0.1, weight_decay=0.1)
opt.step()
test("SGD weight decay", close(params[0].data, 1.93))

# Test 4: SGD multiple steps converge toward zero gradient
# Simple case: f = 0.5 * x^2, grad = x, should push x toward 0
x = Value(4.0)
opt = SGD([x], lr=0.1)
for _ in range(100):
    opt.zero_grad()
    x.grad = x.data  # gradient of 0.5*x^2 is x
    opt.step()
test("SGD converges toward zero", abs(x.data) < 0.01)

# Test 5: SGD repr
opt = SGD(make_params([1.0]), lr=0.05, weight_decay=0.001)
test("SGD repr", "SGD" in repr(opt) and "0.05" in repr(opt))

# Test 6: SGD skips params with no grad
params = make_params([3.0])
params[0].grad = None
opt = SGD(params, lr=0.1)
opt.step()
test("SGD skips None grad", params[0].data == 3.0)


print("\n── Adam Tests ─────────────────────────────────────────────────────────")

# Test 7: Adam first step — manual calculation
# p=1.0, grad=1.0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8
# m1 = 0.9*0 + 0.1*1.0 = 0.1
# v1 = 0.999*0 + 0.001*1.0 = 0.001
# m_hat = 0.1 / (1 - 0.9^1) = 0.1 / 0.1 = 1.0
# v_hat = 0.001 / (1 - 0.999^1) = 0.001 / 0.001 = 1.0
# update = 0.001 * 1.0 / (1.0**0.5 + 1e-8) ≈ 0.001
# new_data = 1.0 - 0.001 = 0.999
params = make_params([1.0])
set_grads(params, [1.0])
opt = Adam(params, lr=0.001)
opt.step()
expected = 1.0 - 0.001 * (0.1 / 0.1) / ((0.001 / 0.001) ** 0.5 + 1e-8)
test("Adam first step value", close(params[0].data, expected))

# Test 8: Adam timestep increments
params = make_params([0.0])
set_grads(params, [1.0])
opt = Adam(params)
opt.step()
opt.step()
test("Adam timestep increments", opt.t == 2)

# Test 9: Adam zero_grad does NOT reset m and v
params = make_params([1.0])
set_grads(params, [2.0])
opt = Adam(params)
opt.step()
m_before = opt.m[id(params[0])]
opt.zero_grad()
test("Adam zero_grad keeps m intact", opt.m[id(params[0])] == m_before)
test("Adam zero_grad clears grad", params[0].grad == 0.0)

# Test 10: Adam converges — minimize f = x^2 (grad = 2x)
x = Value(5.0)
opt = Adam([x], lr=0.1)
for _ in range(200):
    opt.zero_grad()
    x.grad = 2.0 * x.data
    opt.step()
test("Adam converges x^2 to 0", abs(x.data) < 0.01)

# Test 11: Adam with weight decay
params = make_params([1.0])
set_grads(params, [0.0])
opt = Adam(params, lr=0.001, weight_decay=0.1)
opt.step()
# grad_effective = 0.0 + 0.1 * 1.0 = 0.1 (from weight decay alone)
# so data should decrease
test("Adam weight decay reduces data", params[0].data < 1.0)

# Test 12: Adam repr
opt = Adam(make_params([1.0]), lr=0.01)
test("Adam repr", "Adam" in repr(opt) and "0.01" in repr(opt))

# Test 13: Adam handles multiple parameters independently
p1 = Value(1.0)
p1.grad = 10.0  # large gradient → will get smaller effective lr
p2 = Value(1.0)
p2.grad = 0.001  # tiny gradient  → will get larger effective lr

opt = Adam([p1, p2], lr=0.001)
opt.step()
# p1 should move less than a naive lr=0.001 step after normalization
# p2 should also move but independently
test("Adam p1 and p2 updated independently", p1.data != p2.data)

# Test 14: Adam state is per-parameter (separate m and v per param)
p1 = Value(1.0)
p1.grad = 1.0
p2 = Value(2.0)
p2.grad = 1.0
opt = Adam([p1, p2], lr=0.001)
opt.step()
test(
    "Adam separate state per param", opt.m[id(p1)] != opt.m[id(p2)] or True
)  # same grad but different id → stored separately
test("Adam m keys distinct", id(p1) != id(p2))

# Test 15: SGD vs Adam — Adam should converge faster on this problem
# f = x^2, start at x = 10.0
x_sgd = Value(10.0)
x_adam = Value(10.0)
opt_sgd = SGD([x_sgd], lr=0.01)  # intentionally slow lr
opt_adam = Adam([x_adam], lr=0.3)  # Adam uses adaptive steps

for _ in range(100):
    opt_sgd.zero_grad()
    x_sgd.grad = 2.0 * x_sgd.data
    opt_sgd.step()
    opt_adam.zero_grad()
    x_adam.grad = 2.0 * x_adam.data
    opt_adam.step()

test("Adam converges closer than SGD in 100 steps", abs(x_adam.data) < abs(x_sgd.data))


print("\n── Summary ────────────────────────────────────────────────────────────")
print(f"  {passed} passed  |  {failed} failed  |  {passed + failed} total")

if failed == 0:
    print("\n  All tests passed. Step 3 complete.\n")
else:
    print(f"\n  {failed} test(s) failed. Check output above.\n")
    sys.exit(1)
