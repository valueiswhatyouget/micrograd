# Neural Network From Scratch

A working neural network and the automatic-differentiation engine that powers it,
written in pure Python. No PyTorch, no TensorFlow, no numpy in the core. The engine
and the network are about 230 lines together. It trains, with real backpropagation,
on a problem a straight line can't solve, and it gets it 100% right. And every
gradient in the engine is checked against numerical differentiation, so "the
backprop is correct" is something this repo verifies, not something it asks you to
take on trust (see [Proving the gradients](#proving-the-gradients)).

![Decision boundary the network learned](decision_boundary.png)

*The curved boundary the network found on its own. No straight line can separate
these two classes; backpropagation found the shape.*

## Why I built it

Not to compete with the big frameworks. I wanted to understand, end to end, what
"training a model" actually means. The mechanism here (forward pass, loss,
backpropagation, gradient step) is the same one behind the models everyone talks
about. This is the small, readable version, where you can see every piece.

## The files

| File | What it is |
|------|-----------|
| `engine.py` | The autograd engine. A `Value` wraps one number and remembers how it was computed, so `.backward()` walks the graph in reverse and gets the gradients via the chain rule. This is backpropagation. Ops: `+ - * /`, powers, `tanh`, `relu`, `exp`, `log`. |
| `nn.py` | A neural network (Neuron, Layer, MLP) built on top of those `Value`s. Just weighted sums and a nonlinearity, repeated. |
| `train.py` | Builds a two-moons dataset, trains the net, prints the loss falling and accuracy rising, and saves the decision-boundary image. |
| `gradcheck.py` | Verifies every gradient in the engine against numerical differentiation. Runs in CI on every push. |
| `lm.py` | A character-level language model on the same engine: character embeddings, a small network, a softmax over the vocabulary, and a cross-entropy loss. |
| `train_lm.py` | Trains the language model on any text file (or a built-in corpus), samples text as it learns, and checkpoints so you can stop and resume. |

## Run it

```bash
python -m pip install --upgrade pip   # ensure a modern pip
pip install -r requirements.txt   # numpy + matplotlib, for the plot only
python train.py
```

Output:

```
step   0   loss 0.9301   acc 73%
step  40   loss 0.1534   acc 94%
step  80   loss 0.0184   acc 100%
step  99   loss 0.0105   acc 100%

Saved decision_boundary.png
```

The training itself needs no dependencies. numpy and matplotlib are only used to
draw the picture.

## Proving the gradients

The whole point of this repo is that the backprop is *real*. So how do you know it's
*correct*? There's a second, completely independent way to get a gradient: nudge an
input a tiny bit and measure how much the output moves. That's numerical
differentiation, and it needs nothing but the forward pass. If it agrees with what
`.backward()` produced, the chain-rule code is right.

`gradcheck.py` does exactly that for every operation and for randomly-generated
expression graphs:

```bash
python gradcheck.py
```

```
  add                        max rel error 3.04e-10   [ok]
  mul                        max rel error 2.71e-10   [ok]
  div                        max rel error 3.29e-10   [ok]
  tanh                       max rel error 6.95e-11   [ok]
  relu                       max rel error 8.23e-11   [ok]
  exp                        max rel error 1.61e-10   [ok]
  log                        max rel error 1.42e-10   [ok]
  random graphs              max rel error 7.48e-10   [ok]

all gradients verified against numerical differentiation
```

Karpathy's micrograd checks its gradients against PyTorch. This checks them against
the definition of a derivative, no framework, just math. It runs in CI, so a change
that breaks a gradient fails the build.

## What's actually happening

Training is one idea in a loop:

1. **Forward.** Run inputs through the net, measure how wrong it is (the loss).
2. **Backward.** Call `.backward()` to find, for every weight, how much it
   contributed to that wrongness (the gradient).
3. **Step.** Nudge every weight a little in the direction that lowers the loss.

Do that a few dozen times and random starting weights turn into a model that has
learned the shape of the data.

## A language model on the same engine

The same `Value` and the same `.backward()` can train a language model. `lm.py`
builds a character-level one: it embeds each character, feeds the last few characters
through a small network, and predicts the next character with a softmax over the
vocabulary, trained by cross-entropy loss. Same backprop, pointed at text.

```bash
python train_lm.py                      # built-in corpus
python train_lm.py yourtext.txt         # your own text file
python train_lm.py --steps 4000 --hidden 96 --save run.json   # longer, checkpointed
```

Watch it learn (the built-in corpus is a handful of proverbs):

```
step   0   loss 3.37   sample: 'pmiocuiuxgzdqxauj.rhjuuqvokytgrxkfd'
step 100   loss 2.23   sample: 'hpne scne oachedateept lik. rsp cor the fiktie'
step 200   loss 1.66   sample: 'ioan tirs the pige xuate the mar. ches aardpagl'
step 250   loss 1.58   sample: 'aok iuy the sane saurlthe ptiol canwy coxicthen'
```

It starts as noise and, with nothing but gradient descent, finds spaces, the word
"the", and sentence-ending periods. It is pure-Python scalar autograd, so it is slow
and it stays small: it learns the shape of text, not its meaning. `--save` and
`--resume` checkpoint the weights, so you can train in long stretches and pick up
where you left off. The cross-entropy loss is gradient-checked in `gradcheck.py`
alongside every other operation, so the language model rests on the same
proven-correct backprop as the rest of the repo.

## Where to take it next

Swap `tanh` for `relu` in `nn.py` and watch the boundary change. Feed `train_lm.py`
a bigger text file, widen the context window, and let it run longer. The natural next
steps are a proper embedding lookup table, a wider context, and eventually attention:
the same engine, scaled up as far as pure Python can carry it.

Built in the spirit of Andrej Karpathy's `micrograd`, as a way to learn how model
training works from first principles.
