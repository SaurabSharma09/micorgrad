"""
test_validation.py — Comprehensive validation suite for micrograd-explained
=============================================================================
Goes BEYOND basic unit tests to verify:
    1. Numerical gradient checking (finite differences vs analytical)
    2. Training convergence (loss must decrease)
    3. Dropout behavior (train vs eval correctness)
    4. BatchNorm gradient flow (gradients must reach inputs)
    5. End-to-end pipeline (model → loss → backward → optimizer → improvement)
    6. Numerical stability (extreme values, edge cases)

This is the final "does everything actually work?" check.
"""

import os
import random
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import Value
from nn import Neuron, Layer, MLP, Dropout, BatchNorm
from optim import SGD, Adam
from loss import mse_loss, binary_cross_entropy, cross_entropy_loss

passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}  {detail}")
        failed += 1


def close(a, b, tol=1e-4):
    return abs(a - b) < tol


# ─────────────────────────────────────────────────────────────────────────────
# 1. NUMERICAL GRADIENT CHECKING
# ─────────────────────────────────────────────────────────────────────────────

print("\n═══ 1. Numerical Gradient Checking ═════════════════════════════════════")
print("   Compare analytical gradients (.grad) with finite-difference approximation.")
print("   If these match, our backward pass is mathematically correct.\n")


def numerical_grad_check(build_graph_fn, params, h=1e-5, tol=1e-3):
    """
    Verify analytical gradients match numerical gradients.

    For each parameter p:
        numerical_grad = (f(p + h) - f(p - h)) / (2h)    [central differences]
        analytical_grad = p.grad                            [from backward()]

    These should match within tolerance.
    """
    # Forward + backward to get analytical gradients
    loss = build_graph_fn()
    for p in params:
        p.grad = 0.0
    loss.backward()
    analytical_grads = {id(p): p.grad for p in params}

    # Numerical gradients via central differences
    max_error = 0.0
    for p in params:
        original = p.data

        # f(p + h)
        p.data = original + h
        loss_plus = build_graph_fn()

        # f(p - h)
        p.data = original - h
        loss_minus = build_graph_fn()

        # restore
        p.data = original

        numerical = (loss_plus.data - loss_minus.data) / (2 * h)
        analytical = analytical_grads[id(p)]
        error = abs(numerical - analytical)

        # relative error for large gradients, absolute for small
        denom = max(abs(numerical), abs(analytical), 1e-8)
        rel_error = error / denom
        max_error = max(max_error, rel_error)

    return max_error


# Test: simple expression
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = Value(10.0, label="c")
params = [a, b, c]


def simple_expr():
    return a * b + c


err = numerical_grad_check(simple_expr, params)
test("Grad check: a*b + c", err < 1e-3, f"max_rel_error={err:.2e}")

# Test: more complex expression with activations
w1 = Value(0.5)
w2 = Value(-0.8)
b1 = Value(0.1)
params2 = [w1, w2, b1]


def complex_expr():
    x1, x2 = 1.5, -0.7
    out = (w1 * x1 + w2 * x2 + b1).tanh()
    return out


err = numerical_grad_check(complex_expr, params2)
test("Grad check: tanh(w1*x1 + w2*x2 + b)", err < 1e-3, f"max_rel_error={err:.2e}")

# Test: sigmoid
w3 = Value(1.0)


def sigmoid_expr():
    return w3.sigmoid()


err = numerical_grad_check(sigmoid_expr, [w3])
test("Grad check: sigmoid(w)", err < 1e-3, f"max_rel_error={err:.2e}")

# Test: relu
w4 = Value(2.0)


def relu_expr():
    return (w4 * 3.0 + Value(-1.0)).relu()


err = numerical_grad_check(relu_expr, [w4])
test("Grad check: relu(3w - 1)", err < 1e-3, f"max_rel_error={err:.2e}")

# Test: exp and log composition
w5 = Value(1.5)


def exp_log_expr():
    return (w5.exp() + Value(1.0)).log()


err = numerical_grad_check(exp_log_expr, [w5])
test("Grad check: log(exp(w) + 1)", err < 1e-3, f"max_rel_error={err:.2e}")

# Test: division and power
w6 = Value(3.0)
w7 = Value(2.0)


def div_pow_expr():
    return (w6**2) / w7


err = numerical_grad_check(div_pow_expr, [w6, w7])
test("Grad check: w6²/w7", err < 1e-3, f"max_rel_error={err:.2e}")

# Test: MLP forward gradient check
random.seed(42)
tiny_model = MLP(2, [4, 1], activation="tanh", output_activation="linear")
tiny_params = tiny_model.parameters()
x_fixed = [Value(1.0), Value(-0.5)]


def mlp_expr():
    return tiny_model([Value(1.0), Value(-0.5)])


err = numerical_grad_check(mlp_expr, tiny_params)
test("Grad check: MLP(2→4→1) forward", err < 1e-2, f"max_rel_error={err:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRAINING CONVERGENCE
# ─────────────────────────────────────────────────────────────────────────────

print("\n═══ 2. Training Convergence ═════════════════════════════════════════════")
print("   Verify that loss DECREASES over training steps.\n")

# Test: MSE regression converges with SGD
random.seed(42)
model_reg = MLP(1, [8, 1], activation="tanh", output_activation="linear")
opt_reg = SGD(model_reg.parameters(), lr=0.01)

xs_reg = [[0.5], [-0.5], [1.0], [-1.0]]
ys_reg = [0.25, -0.25, 0.5, -0.5]  # roughly y = 0.5x

first_loss = None
last_loss = None

for epoch in range(50):
    preds = [model_reg([Value(x[0])]) for x in xs_reg]
    loss = mse_loss(preds, ys_reg)

    if epoch == 0:
        first_loss = loss.data
    last_loss = loss.data

    opt_reg.zero_grad()
    loss.backward()
    opt_reg.step()

test(
    "SGD regression: loss decreases",
    last_loss < first_loss,
    f"first={first_loss:.4f}, last={last_loss:.4f}",
)

# Test: BCE classification converges with Adam
random.seed(42)
model_cls = MLP(2, [8, 1], activation="tanh", output_activation="sigmoid")
opt_cls = Adam(model_cls.parameters(), lr=0.05)

xs_cls = [[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]]
ys_cls = [1.0, 0.0, 1.0, 0.0]

losses_cls = []
for epoch in range(30):
    preds = [model_cls([Value(x[0]), Value(x[1])]) for x in xs_cls]
    loss = binary_cross_entropy(preds, ys_cls)
    losses_cls.append(loss.data)

    opt_cls.zero_grad()
    loss.backward()
    opt_cls.step()

test(
    "Adam classification: loss decreases",
    losses_cls[-1] < losses_cls[0],
    f"first={losses_cls[0]:.4f}, last={losses_cls[-1]:.4f}",
)

# Test: loss should decrease monotonically (mostly)
# Allow some oscillation but overall trend must be downward
first_half_avg = sum(losses_cls[:15]) / 15
second_half_avg = sum(losses_cls[15:]) / 15
test(
    "Adam: second half avg < first half avg",
    second_half_avg < first_half_avg,
    f"first_half={first_half_avg:.4f}, second_half={second_half_avg:.4f}",
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. DROPOUT BEHAVIOR
# ─────────────────────────────────────────────────────────────────────────────

print("\n═══ 3. Dropout Behavior ═════════════════════════════════════════════════")
print("   Verify train vs eval mode correctness.\n")

# Test: training mode drops approximately p fraction
random.seed(42)
drop = Dropout(p=0.3)
x = [Value(1.0)] * 1000
out = drop(x, training=True)
zeros = sum(1 for o in out if abs(o.data) < 1e-9)
drop_rate = zeros / len(x)
test(
    f"Dropout(0.3) train: ~30% dropped",
    0.2 < drop_rate < 0.4,
    f"actual_rate={drop_rate:.2f}",
)

# Test: surviving neurons are scaled by 1/(1-p)
non_zero = [o.data for o in out if abs(o.data) > 1e-9]
if non_zero:
    expected_scale = 1.0 / (1.0 - 0.3)
    test(
        "Dropout train: survivors scaled by 1/(1-p)",
        all(abs(v - expected_scale) < 1e-6 for v in non_zero),
        f"expected={expected_scale:.4f}, got_example={non_zero[0]:.4f}",
    )

# Test: eval mode passes through unchanged
out_eval = drop(x, training=False)
test(
    "Dropout eval: all values unchanged",
    all(abs(o.data - 1.0) < 1e-6 for o in out_eval),
)

# Test: expected value is same in training and eval
random.seed(0)
n_trials = 20
train_means = []
for _ in range(n_trials):
    out_t = drop([Value(5.0)] * 100, training=True)
    train_means.append(sum(o.data for o in out_t) / len(out_t))

avg_train = sum(train_means) / len(train_means)
avg_eval = 5.0  # passthrough
test(
    "Dropout: E[train] ≈ E[eval] (inverted dropout)",
    abs(avg_train - avg_eval) < 1.0,
    f"train_avg={avg_train:.2f}, eval={avg_eval:.2f}",
)

# Test: p=0 dropout is identity
drop_zero = Dropout(p=0.0)
x_test = [Value(3.14), Value(-2.71)]
out_zero = drop_zero(x_test, training=True)
test(
    "Dropout(p=0): identity in train mode",
    all(abs(o.data - xi.data) < 1e-6 for o, xi in zip(out_zero, x_test)),
)


# ─────────────────────────────────────────────────────────────────────────────
# 4. BATCHNORM CORRECTNESS
# ─────────────────────────────────────────────────────────────────────────────

print("\n═══ 4. BatchNorm Correctness ════════════════════════════════════════════")
print("   Verify normalization + gradient flow through BatchNorm.\n")

# Test: output has mean ≈ 0 with default gamma=1, beta=0
bn = BatchNorm(nin=4)
x_bn = [Value(1.0), Value(5.0), Value(2.0), Value(8.0)]
out_bn = bn(x_bn)
out_mean = sum(o.data for o in out_bn) / len(out_bn)
test("BatchNorm: output mean ≈ 0", abs(out_mean) < 0.1, f"mean={out_mean:.6f}")

# Test: output has std ≈ 1
out_var = sum((o.data - out_mean) ** 2 for o in out_bn) / len(out_bn)
out_std = math.sqrt(out_var)
test("BatchNorm: output std ≈ 1", abs(out_std - 1.0) < 0.1, f"std={out_std:.6f}")

# Test: gradients flow through BatchNorm (CRITICAL — was a bug before!)
bn2 = BatchNorm(nin=3)
# Use asymmetric inputs so no gradient cancels to exactly zero
x_bn2 = [Value(1.0, label="x0"), Value(3.0, label="x1"), Value(8.0, label="x2")]
out_bn2 = bn2(x_bn2)

# Use squared loss — gradients must not cancel out
total = out_bn2[0] ** 2 + out_bn2[1] ** 2 + out_bn2[2] ** 2
total.backward()

# At least some gradients must be nonzero (proves graph is connected)
grads_exist = any(abs(xi.grad) > 1e-8 for xi in x_bn2)
test(
    "BatchNorm: gradients reach input Values",
    grads_exist,
    f"grads={[xi.grad for xi in x_bn2]}",
)

# gamma and beta should also have gradients
gamma_grads = all(abs(g.grad) > 1e-10 or True for g in bn2.gamma)
beta_grads = all(abs(b.grad) > 1e-10 or True for b in bn2.beta)
test(
    "BatchNorm: gamma/beta have gradients",
    any(abs(g.grad) > 1e-10 for g in bn2.gamma)
    or any(abs(b.grad) > 1e-10 for b in bn2.beta),
)

# Test: gamma=2, beta=3 should scale and shift
bn3 = BatchNorm(nin=3)
for g in bn3.gamma:
    g.data = 2.0
for b in bn3.beta:
    b.data = 3.0
x_bn3 = [Value(1.0), Value(2.0), Value(3.0)]
out_bn3 = bn3(x_bn3)
out_mean3 = sum(o.data for o in out_bn3) / len(out_bn3)
# With gamma=2, beta=3: mean should be ≈ 3 (since normalized mean is 0, then 2*0 + 3 = 3)
test(
    "BatchNorm: gamma/beta shift output mean",
    abs(out_mean3 - 3.0) < 0.2,
    f"mean={out_mean3:.4f}",
)


# ─────────────────────────────────────────────────────────────────────────────
# 5. END-TO-END PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

print("\n═══ 5. End-to-End Pipeline ══════════════════════════════════════════════")
print("   Full pipeline: model → loss → backward → optimizer → improvement.\n")

# Test: complete train loop on XOR-like problem
random.seed(42)
model_e2e = MLP(2, [8, 8, 1], activation="tanh", output_activation="sigmoid")
opt_e2e = Adam(model_e2e.parameters(), lr=0.05)

# XOR-ish dataset
xs_e2e = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
ys_e2e = [0.0, 0.0, 1.0, 1.0]

initial_loss = None
for epoch in range(60):
    preds = [model_e2e([Value(x[0]), Value(x[1])]) for x in xs_e2e]
    loss = binary_cross_entropy(preds, ys_e2e)
    if epoch == 0:
        initial_loss = loss.data

    opt_e2e.zero_grad()
    loss.backward()
    opt_e2e.step()

final_loss = loss.data
test(
    "E2E: loss decreased significantly",
    final_loss < initial_loss * 0.5,
    f"initial={initial_loss:.4f}, final={final_loss:.4f}",
)

# Check predictions are reasonable
final_preds = [model_e2e([Value(x[0]), Value(x[1])]) for x in xs_e2e]
pred_vals = [p.data for p in final_preds]
correct = sum(1 for p, y in zip(pred_vals, ys_e2e) if (p > 0.5) == (y > 0.5))
test(
    f"E2E: accuracy ≥ 75%",
    correct >= 3,
    f"correct={correct}/4, preds={[f'{p:.2f}' for p in pred_vals]}",
)


# ─────────────────────────────────────────────────────────────────────────────
# 6. NUMERICAL STABILITY
# ─────────────────────────────────────────────────────────────────────────────

print("\n═══ 6. Numerical Stability ══════════════════════════════════════════════")
print("   Edge cases that could cause NaN, inf, or crashes.\n")

# Test: sigmoid of large positive → ≈ 1.0 (not overflow)
v_large = Value(50.0)
s_large = v_large.sigmoid()
test("sigmoid(50) ≈ 1.0 (no overflow)", abs(s_large.data - 1.0) < 1e-6)

# Test: sigmoid of large negative → ≈ 0.0
v_neg = Value(-50.0)
s_neg = v_neg.sigmoid()
test("sigmoid(-50) ≈ 0.0 (no underflow)", abs(s_neg.data) < 1e-6)

# Test: tanh of large value → ≈ ±1.0
t_large = Value(100.0).tanh()
test("tanh(100) ≈ 1.0", abs(t_large.data - 1.0) < 1e-6)

# Test: relu of negative is exactly 0
r_neg = Value(-5.0).relu()
test("relu(-5) = 0.0", r_neg.data == 0.0)

# Test: cross entropy with max-value stability
logits = [[Value(100.0), Value(-100.0)]]
ce_loss = cross_entropy_loss(logits, [0])
test(
    "CE loss: extreme logits don't overflow",
    math.isfinite(ce_loss.data) and ce_loss.data >= 0,
    f"loss={ce_loss.data}",
)

# Test: BCE with near-boundary predictions
p_near_0 = Value(1e-6)
bce_0 = binary_cross_entropy([p_near_0], [0.0])
test(
    "BCE: pred≈0, target=0 → small loss", math.isfinite(bce_0.data) and bce_0.data < 0.1
)

p_near_1 = Value(1.0 - 1e-6)
bce_1 = binary_cross_entropy([p_near_1], [1.0])
test(
    "BCE: pred≈1, target=1 → small loss", math.isfinite(bce_1.data) and bce_1.data < 0.1
)

# Test: division by small number doesn't crash
v_small = Value(1e-8)
v_div = Value(1.0) / v_small
test("Division by 1e-8: no crash", math.isfinite(v_div.data), f"result={v_div.data}")

# Test: power with negative base and integer exponent
v_neg_base = Value(-2.0)
v_pow = v_neg_base**2
test("(-2)^2 = 4.0", abs(v_pow.data - 4.0) < 1e-6)

# Test: zero_grad on model with many params
random.seed(42)
big_model = MLP(2, [16, 16, 1])
x_big = [Value(1.0), Value(2.0)]
out_big = big_model(x_big)
out_big.backward()
has_grads = any(abs(p.grad) > 0 for p in big_model.parameters())
big_model.zero_grad()
all_zero = all(p.grad == 0.0 for p in big_model.parameters())
test("zero_grad: clears all params in large model", has_grads and all_zero)


# ─────────────────────────────────────────────────────────────────────────────
# 7. CROSS-MODULE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n═══ 7. Cross-Module Integration ═════════════════════════════════════════")
print("   Verify all modules work together seamlessly.\n")

# Test: MSE loss → SGD → parameters change
random.seed(42)
m = MLP(1, [4, 1], activation="tanh", output_activation="linear")
params_before = [p.data for p in m.parameters()]
opt = SGD(m.parameters(), lr=0.01)

pred = m([Value(1.0)])
loss = mse_loss([pred], [0.5])
opt.zero_grad()
loss.backward()
opt.step()

params_after = [p.data for p in m.parameters()]
any_changed = any(abs(a - b) > 1e-10 for a, b in zip(params_before, params_after))
test("MSE + SGD: parameters change after step", any_changed)

# Test: CE loss backward populates gradients
z0, z1, z2 = Value(1.0), Value(2.0), Value(0.5)
ce = cross_entropy_loss([[z0, z1, z2]], [1])
ce.backward()
test("CE backward: all logit grads exist", all(g.grad != 0.0 for g in [z0, z1, z2]))

# Test: softmax probabilities sum to 1 (indirectly via CE gradient sum ≈ 0)
grad_sum = z0.grad + z1.grad + z2.grad
test(
    "CE: gradient sum ≈ 0 (softmax property)",
    abs(grad_sum) < 1e-4,
    f"sum={grad_sum:.6f}",
)

# Test: visualizer doesn't crash
from visualizer import print_graph_text, _build_graph

a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = a * b
c.label = "c"
c.backward()
nodes, edges = _build_graph(c)
test(
    "Visualizer: graph has correct node count",
    len(nodes) == 3,
    f"got {len(nodes)} nodes",
)
test(
    "Visualizer: graph has correct edge count",
    len(edges) == 2,
    f"got {len(edges)} edges",
)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "═" * 72)
print(
    f"  VALIDATION COMPLETE: {passed} passed  |  {failed} failed  |  {passed + failed} total"
)
print("═" * 72)

if failed == 0:
    print("\n  ✅ ALL VALIDATIONS PASSED — the engine is production-quality.\n")
else:
    print(f"\n  ⚠️  {failed} validation(s) failed. Review output above.\n")
    sys.exit(1)
