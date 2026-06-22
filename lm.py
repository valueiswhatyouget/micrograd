"""
lm.py: a character-level language model on the from-scratch engine.

Same Value and the same .backward() as the rest of the repo, now pointed at
text instead of a toy dataset. The model reads the last few characters and
predicts the next one. Train it on any text and it learns that text's patterns,
then generates new text one character at a time.

It is small and pure Python (every number is a scalar Value), so it is slow and
it will not write essays. But it is a real language model: character embeddings,
a neural net, a softmax over the vocabulary, a cross-entropy loss, and gradient
descent. The same machinery, scaled down far enough that you can read all of it.
"""

import json
import math
import random

from engine import Value
from nn import Layer, Module


class CharDataset:
    """Turns raw text into (context, next-character) training pairs.

    The vocabulary is just the distinct characters in the text. Each example is
    a fixed-length window of characters and the character that follows it.
    """

    def __init__(self, text, block_size=3):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
        self.block_size = block_size
        self._ids = [self.stoi[c] for c in text]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def examples(self):
        """Yield (context_ids, target_id) sliding over the text.

        The start is padded with the first character so the very first real
        characters still have a full context to predict from.
        """
        seq = [self._ids[0]] * self.block_size + self._ids
        for i in range(len(self._ids)):
            yield seq[i:i + self.block_size], seq[i + self.block_size]


class CharLM(Module):
    """Embed each character, run the context through an MLP, predict the next."""

    def __init__(self, vocab_size, block_size, embed_dim=8, hidden=64):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim
        # one small trainable vector per character
        scale = embed_dim ** -0.5
        self.emb = [
            [Value(random.uniform(-1, 1) * scale) for _ in range(embed_dim)]
            for _ in range(vocab_size)
        ]
        self.hidden = Layer(block_size * embed_dim, hidden, nonlin=True)
        self.out = Layer(hidden, vocab_size, nonlin=False)   # linear: raw logits

    def logits(self, context):
        """Forward pass: context of char ids -> one logit per vocabulary char."""
        x = []
        for idx in context:
            x.extend(self.emb[idx])     # look up (and keep differentiable) the embeddings
        return self.out(self.hidden(x))

    def parameters(self):
        emb_params = [v for row in self.emb for v in row]
        return emb_params + self.hidden.parameters() + self.out.parameters()


def cross_entropy(logits, target):
    """Negative log-likelihood of the target character under a softmax.

    Subtracting the max logit first is the standard trick to keep exp() from
    overflowing; it is a constant shift, so it does not change the gradient.
    """
    shift = max(logit.data for logit in logits)
    exps = [(logit - shift).exp() for logit in logits]
    total = sum(exps, Value(0.0))
    return -(exps[target] / total).log()


def generate(model, dataset, n_chars=300, seed=None):
    """Sample text from the model, one character at a time."""
    rng = random.Random(seed)
    context = [0] * model.block_size
    out = []
    for _ in range(n_chars):
        logits = model.logits(context)
        # softmax in plain floats; this is sampling, no gradient needed
        shift = max(logit.data for logit in logits)
        probs = [math.exp(logit.data - shift) for logit in logits]
        total = sum(probs)
        probs = [p / total for p in probs]
        idx = _sample_index(probs, rng)
        out.append(idx)
        context = context[1:] + [idx]
    return dataset.decode(out)


def _sample_index(probs, rng):
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probs) - 1


def save_checkpoint(path, model, text):
    """Write the corpus, the model shape, and every weight to a JSON file."""
    with open(path, "w") as fh:
        json.dump({
            "text": text,
            "block_size": model.block_size,
            "embed_dim": model.embed_dim,
            "hidden": len(model.hidden.neurons),
            "params": [p.data for p in model.parameters()],
        }, fh)


def load_checkpoint(path):
    """Rebuild the dataset and a trained model from a checkpoint file."""
    with open(path) as fh:
        ckpt = json.load(fh)
    data = CharDataset(ckpt["text"], ckpt["block_size"])
    model = CharLM(data.vocab_size, ckpt["block_size"], ckpt["embed_dim"], ckpt["hidden"])
    for param, value in zip(model.parameters(), ckpt["params"]):
        param.data = value
    return model, data
