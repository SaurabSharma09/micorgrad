import sys

sys.stdout.reconfigure(encoding="utf-8")

from engine import Value
from nn import MLP, Dropout, BatchNorm
from loss import mse_loss, binary_cross_entropy
from optim import SGD, Adam
from visualizer import draw_graph, print_graph_text

print("=" * 60)
print("  micrograd-explained — Live Demo")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# IMPACT 1: engine.py (autograd)
# ─────────────────────────────────────────────────────────────
print("\n--- IMPACT 1: engine.py (autograd) ---")
print("WITHOUT engine.py:")
print("  a = 2.0, b = -3.0")
print("  c = a * b = -6.0")
print("  gradient? impossible. no tracking.")

print("\nWITH engine.py:")
a = Value(2.0, label="a")
b = Value(-3.0, label="b")
c = a * b
c.label = "c"
c.backward()
print(f"  a = {a}")
print(f"  b = {b}")
print(f"  c = {c}")
print("  gradients computed automatically via chain rule")

# ─────────────────────────────────────────────────────────────
# IMPACT 2: activations
# ─────────────────────────────────────────────────────────────
print("\n--- IMPACT 2: relu and sigmoid (new activations) ---")

x = Value(2.0)
print(f"\n  tanh(2.0)    = {x.tanh().data:.4f}")

x2 = Value(2.0)
print(f"  relu(2.0)    = {x2.relu().data:.4f}")

x3 = Value(2.0)
print(f"  sigmoid(2.0) = {x3.sigmoid().data:.4f}")

x4 = Value(-2.0)
print(f"\n  relu(-2.0)   = {x4.relu().data:.4f}")

# ─────────────────────────────────────────────────────────────
# IMPACT 3: Dropout
# ─────────────────────────────────────────────────────────────
print("\n--- IMPACT 3: Dropout (new layer) ---")

drop = Dropout(p=0.5)
inputs = [Value(1.0) for _ in range(10)]

training_out = drop(inputs, training=True)
zeros = sum(1 for o in training_out if abs(o.data) < 1e-9)
print(f"\n  Training mode:  {zeros}/10 neurons zeroed randomly")

inference_out = drop(inputs, training=False)
print(f"  Inference mode: all neurons active")

# ─────────────────────────────────────────────────────────────
# IMPACT 4: BatchNorm
# ─────────────────────────────────────────────────────────────
print("\n--- IMPACT 4: BatchNorm (new layer) ---")

bn = BatchNorm(nin=4)
raw = [Value(10.0), Value(200.0), Value(0.5), Value(50.0)]
normalized = bn(raw)

vals = [round(o.data, 4) for o in normalized]
mean = sum(o.data for o in normalized) / 4

print(f"\n  Raw inputs:        {[o.data for o in raw]}")
print(f"  After BatchNorm:   {vals}")
print(f"  Mean after norm:   {mean:.6f}")

# ─────────────────────────────────────────────────────────────
# IMPACT 5: Optimizers
# ─────────────────────────────────────────────────────────────
print("\n--- IMPACT 5: Optimizers (SGD vs Adam) ---")

from engine import Value as V

x_sgd = V(5.0)
opt_sgd = SGD([x_sgd], lr=0.1)

for _ in range(50):
    opt_sgd.zero_grad()
    x_sgd.grad = 2.0 * x_sgd.data
    opt_sgd.step()

x_adam = V(5.0)
opt_adam = Adam([x_adam], lr=0.1)

for _ in range(50):
    opt_adam.zero_grad()
    x_adam.grad = 2.0 * x_adam.data
    opt_adam.step()

print(f"\n  SGD final x:   {x_sgd.data:.6f}")
print(f"  Adam final x:  {x_adam.data:.6f}")

# ─────────────────────────────────────────────────────────────
# IMPACT 6: Loss functions (FIXED)
# ─────────────────────────────────────────────────────────────
print("\n--- IMPACT 6: Loss functions ---")

p1 = Value(0.9)
loss_bce = binary_cross_entropy([p1], [1.0])
print(f"\n  BCE(0.9,1): {loss_bce.data:.4f}")

p2 = Value(0.1)
loss_bce2 = binary_cross_entropy([p2], [1.0])
print(f"  BCE(0.1,1): {loss_bce2.data:.4f}")

# ✅ FIX: use sigmoid
p3 = Value(1.0).sigmoid()
p4 = Value(2.0).sigmoid()
p5 = Value(0.5).sigmoid()

loss_ce = binary_cross_entropy([p3, p4, p5], [1.0, 1.0, 0.0])
print(f"  BCE multi: {loss_ce.data:.4f}")

# ─────────────────────────────────────────────────────────────
# IMPACT 7: Full training
# ─────────────────────────────────────────────────────────────
print("\n--- IMPACT 7: Full MLP training ---")

model = MLP(2, [4, 4, 1], activation="tanh", output_activation="tanh")
optimizer = Adam(model.parameters(), lr=0.05)

xs = [
    [Value(2.0), Value(3.0)],
    [Value(-1.0), Value(-1.0)],
    [Value(0.5), Value(1.0)],
    [Value(1.0), Value(-2.0)],
]
ys = [1.0, -1.0, 1.0, -1.0]

for step in range(60):
    preds = [model(x) for x in xs]
    loss = mse_loss(preds, ys)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"step {step} loss={loss.data:.4f}")

print(f"\nFinal loss: {loss.data:.4f}")

# ─────────────────────────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────────────────────────
print("\n--- GRAPH ---")

x1 = Value(1.0, label="x1")
x2 = Value(2.0, label="x2")

small_model = MLP(2, [2, 1])
out = small_model([x1, x2])
target = Value(0.5)

loss = (out - target) ** 2
loss.backward()

print_graph_text(loss)
draw_graph(loss, filename="graph", format="svg")

print("\nSaved graph.svg")
