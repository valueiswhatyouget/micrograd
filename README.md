# Neural Network From Scratch

A working neural network and the automatic-differentiation engine that powers it,
written in pure Python. No PyTorch, no TensorFlow, no numpy in the core. About 200
lines total. It trains, with real backpropagation, on a problem a straight line
can't solve, and it gets it 100% right.

![Decision boundary the network learned](decision_boundary.png)

*The curved boundary the network found on its own. No straight line can separate
these two classes; backpropagation found the shape.*

## Why I built it

Not to compete with the big frameworks. I wanted to understand, end to end, what
"training a model" actually means. The mechanism here (forward pass, loss,
backpropagation, gradient step) is the same one behind the models everyone talks
about. This is the small, readable version, where you can see every piece.

## The three files

| File | What it is |
|------|-----------|
| `engine.py` | The autograd engine. A `Value` wraps one number and remembers how it was computed, so `.backward()` walks the graph in reverse and gets the gradients via the chain rule. This is backpropagation. |
| `nn.py` | A neural network (Neuron, Layer, MLP) built on top of those `Value`s. Just weighted sums and a nonlinearity, repeated. |
| `train.py` | Builds a two-moons dataset, trains the net, prints the loss falling and accuracy rising, and saves the decision-boundary image. |

## Run it

```bash
python -m pip install --upgrade pip   # ensure a modern pip
pip install -r requirements.txt   # numpy + matplotlib, for the plot only
python train.py
```

Output:

```
step   0   loss 0.77   acc 72%
step  20   loss 0.06   acc 98%
step  49   loss 0.02   acc 100%

Saved decision_boundary.png
```

The training itself needs no dependencies. numpy and matplotlib are only used to
draw the picture.

## What's actually happening

Training is one idea in a loop:

1. **Forward.** Run inputs through the net, measure how wrong it is (the loss).
2. **Backward.** Call `.backward()` to find, for every weight, how much it
   contributed to that wrongness (the gradient).
3. **Step.** Nudge every weight a little in the direction that lowers the loss.

Do that a few dozen times and random starting weights turn into a model that has
learned the shape of the data.

## Where to take it next

Swap `tanh` for `relu` in `nn.py` and watch the boundary change. Make the data
harder (more noise, a spiral instead of moons). The natural next step on this track
is to grow the engine into a tiny character-level language model, which is the same
idea pointed at text.

Built in the spirit of Andrej Karpathy's `micrograd`, as a way to learn how model
training works from first principles.
