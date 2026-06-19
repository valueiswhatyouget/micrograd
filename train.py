"""
train.py — train OUR neural net, with OUR backprop, on a real problem.

The task: a "two moons" dataset. Two interleaving crescent-shaped classes
that a straight line cannot separate. The network has to learn a curved
boundary, which it can only do if backprop is actually working.

Run:  python train.py
Out:  loss printed each step, final accuracy, and decision_boundary.png
"""

import math
import random
from nn import MLP

random.seed(1337)


# ---- 1. make the dataset (no sklearn — built by hand) ------------------
def make_moons(n=100, noise=0.1):
    X, y = [], []
    for i in range(n):
        # top moon (label -1) and bottom moon (label +1)
        if i < n // 2:
            t = math.pi * (i / (n // 2))
            px, py = math.cos(t), math.sin(t)
            label = -1.0
        else:
            t = math.pi * ((i - n // 2) / (n // 2))
            px, py = 1 - math.cos(t), 0.5 - math.sin(t)
            label = 1.0
        px += random.uniform(-noise, noise)
        py += random.uniform(-noise, noise)
        X.append([px, py])
        y.append(label)
    return X, y


X, y = make_moons(n=100, noise=0.1)


# ---- 2. the model: 2 inputs -> 16 -> 16 -> 1 output --------------------
model = MLP(2, [16, 16, 1])
print(f"Model has {len(model.parameters())} parameters\n")


# ---- 3. loss: how wrong are we? ----------------------------------------
def loss():
    # max-margin "svm" loss + a little L2 regularization
    preds = [model(x) for x in X]
    losses = [(1 + -yi * pi).relu() for yi, pi in zip(y, preds)]
    data_loss = sum(losses) * (1.0 / len(losses))
    reg = 1e-4 * sum((p * p for p in model.parameters()), start=preds[0] * 0.0)
    total = data_loss + reg

    # accuracy (just for reporting, not used in gradients)
    acc = [(yi > 0) == (pi.data > 0) for yi, pi in zip(y, preds)]
    return total, sum(acc) / len(acc)


# ---- 4. the training loop ----------------------------------------------
# This is the entire idea of "learning":
#   forward  -> compute loss
#   backward -> compute gradient of loss w.r.t. every weight
#   step     -> nudge each weight a little against its gradient
EPOCHS = 50
for k in range(EPOCHS):
    total_loss, acc = loss()

    # reset gradients, then backpropagate
    for p in model.parameters():
        p.grad = 0.0
    total_loss.backward()

    # gradient descent with a simple decaying learning rate
    lr = 1.0 - 0.9 * k / EPOCHS
    for p in model.parameters():
        p.data -= lr * p.grad

    if k % 10 == 0 or k == EPOCHS - 1:
        print(f"step {k:3d}   loss {total_loss.data:.4f}   acc {acc*100:.0f}%")


# ---- 5. draw what the network learned ----------------------------------
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = 0.12
    xs = [p[0] for p in X]; ys = [p[1] for p in X]
    x_min, x_max = min(xs) - 0.5, max(xs) + 0.5
    y_min, y_max = min(ys) - 0.5, max(ys) + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    grid = [[xx.ravel()[i], yy.ravel()[i]] for i in range(xx.size)]
    Z = [model(pt).data > 0 for pt in grid]
    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.35)
    cols = ["#c0392b" if yi < 0 else "#2471a3" for yi in y]
    plt.scatter(xs, ys, c=cols, s=22, edgecolors="white", linewidths=0.5)
    plt.title("Decision boundary my net learned (from-scratch backprop)")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=130)
    print("\nSaved decision_boundary.png")
except ImportError:
    print("\n(Install numpy + matplotlib to also save the picture.)")
