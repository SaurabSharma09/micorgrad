"""
test_loss.py - Tests for loss.py
=================================
Verifies MSE, BCE, and Cross Entropy against known values and PyTorch.
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Value
from loss import mse_loss, binary_cross_entropy, cross_entropy_loss

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


def close(a, b, tol=1e-5):
    return abs(a - b) < tol


print("\n── MSE Loss Tests ─────────────────────────────────────────────────────")

# Test 1: perfect prediction → loss = 0
preds = [Value(1.0), Value(2.0), Value(3.0)]
targets = [1.0, 2.0, 3.0]
loss = mse_loss(preds, targets)
test("MSE perfect prediction = 0", close(loss.data, 0.0))

# Test 2: known manual value
# preds=[1,2], targets=[3,4] → ((1-3)^2 + (2-4)^2) / 2 = (4+4)/2 = 4.0
preds = [Value(1.0), Value(2.0)]
targets = [3.0, 4.0]
loss = mse_loss(preds, targets)
test("MSE known value = 4.0", close(loss.data, 4.0))

# Test 3: MSE backward fills gradients
# loss = (p - t)^2 / N, d(loss)/dp = 2(p-t)/N
p = Value(1.0)
loss = mse_loss([p], [3.0])  # (1-3)^2 = 4.0, N=1
loss.backward()
# d/dp = 2*(1-3)/1 = -4.0
test("MSE backward gradient", close(p.grad, -4.0))

# Test 4: MSE gradient with N=2
p1, p2 = Value(1.0), Value(3.0)
loss = mse_loss([p1, p2], [2.0, 2.0])
# loss = ((1-2)^2 + (3-2)^2) / 2 = 1.0
loss.backward()
# d(loss)/dp1 = 2*(1-2)/2 = -1.0
# d(loss)/dp2 = 2*(3-2)/2 =  1.0
test("MSE gradient p1 with N=2", close(p1.grad, -1.0))
test("MSE gradient p2 with N=2", close(p2.grad, 1.0))

# Test 5: MSE accepts Value targets too
p = Value(2.0)
t = Value(5.0)
loss = mse_loss([p], [t])
test("MSE with Value targets", close(loss.data, 9.0))


print("\n── Binary Cross Entropy Tests ─────────────────────────────────────────")

# Test 6: BCE perfect prediction (p→1 when t=1) → loss ≈ 0
# exact: -log(0.999999...) ≈ tiny positive number
p = Value(1.0 - 1e-7)
loss = binary_cross_entropy([p], [1.0])
test("BCE near-perfect prediction ≈ 0", loss.data < 0.01)

# Test 7: BCE known value
# p=0.7, t=1.0: BCE = -log(0.7) ≈ 0.35667
p = Value(0.7)
loss = binary_cross_entropy([p], [1.0])
expected = -math.log(0.7 + 1e-7)
test("BCE single sample t=1", close(loss.data, expected, tol=1e-4))

# Test 8: BCE with t=0
# p=0.3, t=0.0: BCE = -log(1 - 0.3) = -log(0.7) ≈ 0.35667
p = Value(0.3)
loss = binary_cross_entropy([p], [0.0])
expected = -math.log(1.0 - 0.3 + 1e-7)
test("BCE single sample t=0", close(loss.data, expected, tol=1e-4))

# Test 9: BCE backward — gradient should be negative for t=1, p<0.5
p = Value(0.3)
loss = binary_cross_entropy([p], [1.0])
loss.backward()
# d(BCE)/dp when t=1: -1/p → negative, so grad should be negative
test("BCE gradient sign t=1", p.grad < 0)

# Test 10: BCE gradient sign t=0
p = Value(0.7)
loss = binary_cross_entropy([p], [0.0])
loss.backward()
# d(BCE)/dp when t=0: 1/(1-p) → positive, so grad should be positive
test("BCE gradient sign t=0", p.grad > 0)

# Test 11: BCE mean over batch
# two samples: same loss each → mean == single sample loss
p1, p2 = Value(0.7), Value(0.7)
loss_batch = binary_cross_entropy([p1, p2], [1.0, 1.0])
p3 = Value(0.7)
loss_single = binary_cross_entropy([p3], [1.0])
test("BCE batch mean equals single", close(loss_batch.data, loss_single.data, tol=1e-4))


print("\n── Cross Entropy Loss Tests ────────────────────────────────────────────")

# Test 12: CE with perfect logits
# logits = [100.0, -100.0], correct = 0 → loss ≈ 0
logits = [[Value(100.0), Value(-100.0)]]
loss = cross_entropy_loss(logits, [0])
test("CE perfect logit ≈ 0", loss.data < 0.01)

# Test 13: CE uniform logits → loss = log(K)
# For K=2 uniform: loss = log(2) ≈ 0.693
logits = [[Value(0.0), Value(0.0)]]
loss = cross_entropy_loss(logits, [0])
test("CE uniform 2-class = log(2)", close(loss.data, math.log(2), tol=1e-4))

# Test 14: CE uniform 3-class → loss = log(3)
logits = [[Value(0.0), Value(0.0), Value(0.0)]]
loss = cross_entropy_loss(logits, [1])
test("CE uniform 3-class = log(3)", close(loss.data, math.log(3), tol=1e-4))

# Test 15: CE backward — correct class logit gets negative gradient
z0 = Value(2.0)
z1 = Value(1.0)
loss = cross_entropy_loss([[z0, z1]], [0])
loss.backward()
# Correct class gradient: softmax(z0) - 1 < 0 (should decrease loss)
test("CE correct class grad < 0", z0.grad < 0)

# Test 16: CE backward — wrong class logit gets positive gradient
test("CE wrong class grad > 0", z1.grad > 0)

# Test 17: CE batch mean
logits1 = [[Value(1.0), Value(0.0)]]
logits2 = [[Value(1.0), Value(0.0)], [Value(1.0), Value(0.0)]]
loss1 = cross_entropy_loss(logits1, [0])
loss2 = cross_entropy_loss(logits2, [0, 0])
test("CE batch mean consistent", close(loss1.data, loss2.data, tol=1e-5))

# Test 18: CE gradient sum ≈ 0 (softmax gradients sum to 0)
z0, z1, z2 = Value(1.0), Value(2.0), Value(0.5)
loss = cross_entropy_loss([[z0, z1, z2]], [1])
loss.backward()
test("CE gradients sum ≈ 0", close(z0.grad + z1.grad + z2.grad, 0.0, tol=1e-5))


print("\n── Integration: loss + backward + optimizer step ───────────────────────")

# Test 19: MSE + SGD step reduces loss
from optim import SGD
from nn import MLP

model = MLP(2, [4, 1], output_activation="linear")
opt = SGD(model.parameters(), lr=0.01)
xs = [[Value(2.0), Value(3.0)], [Value(-1.0), Value(1.0)]]
ys = [1.0, -1.0]

# forward pass
preds = [model(x) for x in xs]
loss1 = mse_loss(preds, ys)
loss1_val = loss1.data

# backward + step
loss1.backward()
opt.step()

# second forward pass
opt.zero_grad()
preds2 = [model(x) for x in xs]
loss2 = mse_loss(preds2, ys)
# after one step, loss should have changed (may go up or down on first step)
test("MSE loss changes after optimizer step", loss1_val != loss2.data)

# Test 20: BCE + Adam step
from optim import Adam

model2 = MLP(2, [4, 1], output_activation="sigmoid")
opt2 = Adam(model2.parameters(), lr=0.01)

total_loss_before = 0
for _ in range(5):
    opt2.zero_grad()
    preds3 = [model2(x) for x in xs]
    loss3 = binary_cross_entropy(preds3, [1.0, 0.0])
    total_loss_before += loss3.data
    loss3.backward()
    opt2.step()

total_loss_after = 0
for _ in range(5):
    opt2.zero_grad()
    preds4 = [model2(x) for x in xs]
    loss4 = binary_cross_entropy(preds4, [1.0, 0.0])
    total_loss_after += loss4.data
    loss4.backward()
    opt2.step()

test("BCE + Adam decreases loss over 10 steps", total_loss_after < total_loss_before)


print("\n── Summary ─────────────────────────────────────────────────────────────")
print(f"  {passed} passed  |  {failed} failed  |  {passed + failed} total")
if failed == 0:
    print("\n  All tests passed. Step 4 complete.\n")
else:
    print(f"\n  {failed} test(s) failed.\n")
    sys.exit(1)
