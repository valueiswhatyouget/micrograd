"""
train_lm.py: train the character-level language model and generate text.

    python train_lm.py                  # train on the built-in corpus
    python train_lm.py mytext.txt       # train on your own text file
    python train_lm.py --steps 4000 --hidden 96

Each step samples a small batch of (context, next-char) pairs, sums their
cross-entropy loss, backpropagates through the whole batch at once, and nudges
every parameter (embeddings included) down the gradient. Every so often it
prints the loss and a sample so you can watch the text get less random.

It is pure-Python scalar autograd, so it is slow. Keep the corpus and the model
small, or be patient. The point is to see a language model learn from nothing
but the forward and backward passes in this repo.
"""

import argparse
import random

from engine import Value
from lm import (
    CharDataset,
    CharLM,
    cross_entropy,
    generate,
    load_checkpoint,
    save_checkpoint,
)

# A small built-in corpus so the script runs with no arguments. Short, repetitive
# text learns fastest at this scale; swap in your own file for something richer.
DEFAULT_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "a stitch in time saves nine. "
    "the early bird catches the worm. "
    "all that glitters is not gold. "
    "actions speak louder than words. "
    "practice makes perfect. "
    "better late than never. "
    "the pen is mightier than the sword. "
) * 6


def batch_loss(model, examples, batch, rng):
    picks = [rng.choice(examples) for _ in range(batch)]
    losses = [cross_entropy(model.logits(ctx), target) for ctx, target in picks]
    return sum(losses, Value(0.0)) * (1.0 / batch)


def main():
    p = argparse.ArgumentParser(description="Train a character-level language model.")
    p.add_argument("textfile", nargs="?", help="text file to train on (default: built-in corpus)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--block-size", type=int, default=3, help="characters of context")
    p.add_argument("--embed", type=int, default=8, help="embedding dimension")
    p.add_argument("--hidden", type=int, default=64, help="hidden-layer width")
    p.add_argument("--batch", type=int, default=16, help="examples per step")
    p.add_argument("--lr", type=float, default=0.5, help="learning rate")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sample-every", type=int, default=200)
    p.add_argument("--sample-len", type=int, default=200)
    p.add_argument("--save", help="checkpoint file to write periodically")
    p.add_argument("--resume", help="checkpoint file to continue training from")
    args = p.parse_args()

    random.seed(args.seed)
    if args.resume:
        model, data = load_checkpoint(args.resume)
        text = data.decode(data._ids)
        print(f"resumed from {args.resume}")
    else:
        if args.textfile:
            with open(args.textfile) as fh:
                text = fh.read()
        else:
            text = DEFAULT_TEXT
        data = CharDataset(text, block_size=args.block_size)
        model = CharLM(data.vocab_size, args.block_size, args.embed, args.hidden)

    examples = list(data.examples())
    rng = random.Random(args.seed)

    print(f"corpus: {len(text)} chars, vocab {data.vocab_size}, {len(examples)} examples")
    print(f"model:  {len(model.parameters())} parameters "
          f"(block {model.block_size}, embed {model.embed_dim}, "
          f"hidden {len(model.hidden.neurons)})\n")

    for step in range(args.steps):
        loss = batch_loss(model, examples, args.batch, rng)
        model.zero_grad()
        loss.backward()
        lr = args.lr * (1.0 - 0.9 * step / args.steps)   # gentle decay
        for param in model.parameters():
            param.data -= lr * param.grad

        if step % args.sample_every == 0 or step == args.steps - 1:
            sample = generate(model, data, n_chars=args.sample_len, seed=args.seed + step)
            print(f"step {step:5d}   loss {loss.data:.4f}   sample: {sample!r}")
            if args.save:
                save_checkpoint(args.save, model, text)

    if args.save:
        save_checkpoint(args.save, model, text)
        print(f"\nsaved checkpoint to {args.save}")
    print("\nfinal sample:")
    print(generate(model, data, n_chars=max(args.sample_len, 300), seed=args.seed))


if __name__ == "__main__":
    main()
