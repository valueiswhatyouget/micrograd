"""
gradcheck.py: prove the engine's gradients are correct.

Backpropagation gives us gradients analytically, by applying the chain rule
over the graph (that is all engine.py does). There is a second, completely
independent way to get a gradient: nudge an input by a tiny amount and measure
how much the output moves. That is numerical differentiation, and it needs
nothing but the forward pass.

If the two agree, the backward rules are right. This checks every operation,
and randomly-built expression graphs, against central finite differences. It
runs in CI on every push, so "the gradients are correct" is something this repo
verifies rather than asks you to take on faith.
"""

import random

from engine import Value

# A fixed seed makes the whole check deterministic: the same inputs, and so the
# same error numbers, every run and on every machine.
random.seed(1337)


def max_rel_error(make_fn, n_inputs, sampler, trials=300, eps=1e-6):
    """Largest relative error between analytic and numerical gradients.

    make_fn: returns the function for one trial. It must be deterministic once
             returned, so the analytic pass and both numerical passes evaluate
             the *same* graph (otherwise finite differences are meaningless).
    sampler: returns one random input value, drawn from a region where the op
             is smooth (e.g. positive numbers for log).
    """
    worst = 0.0
    for _ in range(trials):
        fn = make_fn()
        xs = [sampler() for _ in range(n_inputs)]

        # analytic gradient, straight from backprop
        inputs = [Value(x) for x in xs]
        fn(inputs).backward()
        analytic = [v.grad for v in inputs]

        # numerical gradient, one input at a time: (f(x+eps) - f(x-eps)) / 2eps
        for k in range(n_inputs):
            hi, lo = xs[:], xs[:]
            hi[k] += eps
            lo[k] -= eps
            numeric = (fn([Value(v) for v in hi]).data
                       - fn([Value(v) for v in lo]).data) / (2 * eps)
            denom = max(1.0, abs(numeric))
            worst = max(worst, abs(analytic[k] - numeric) / denom)
    return worst


def check(name, make_fn, n_inputs, sampler, tol=1e-5):
    err = max_rel_error(make_fn, n_inputs, sampler)
    ok = err < tol
    print(f"  {name:26s} max rel error {err:.2e}   [{'ok' if ok else 'FAIL'}]")
    if not ok:
        raise AssertionError(f"{name}: gradient error {err:.2e} exceeds {tol:.0e}")


# Samplers that keep inputs inside each op's well-defined, smooth region.
def real_line():
    return random.uniform(-2.0, 2.0)

def positive():            # log, division, fractional powers
    return random.uniform(0.1, 3.0)

def off_the_kink():        # relu is non-differentiable exactly at 0; stay clear
    x = random.uniform(0.2, 2.0)
    return x if random.random() < 0.5 else -x


def make_random_graph(n_inputs):
    """Return a deterministic random expression over n_inputs.

    The op sequence is chosen once, here, so the returned function always builds
    the same graph. Each step folds in one input; the square and exp paths act
    on a single bounded input rather than the running total, so the value can't
    compound into an overflow while still touching every backward rule.
    """
    ops = [random.choice(("add", "mul", "tanh", "square", "exp"))
           for _ in range(n_inputs - 1)]

    def fn(inputs):
        v = inputs[0]
        for x, op in zip(inputs[1:], ops):
            if op == "add":
                v = v + x
            elif op == "mul":
                v = (v * x).tanh()        # squash the product to stay bounded
            elif op == "tanh":
                v = (v + x).tanh()
            elif op == "square":
                v = v + x ** 2            # x is a bounded leaf, so x**2 is small
            elif op == "exp":
                v = v + x.tanh().exp()    # exp of a value in (-1, 1), never large
        return v

    return fn


def static(fn):
    """Wrap a fixed function as a make_fn factory."""
    return lambda: fn


def main():
    print("checking analytic gradients against central finite differences\n")
    check("add",                static(lambda v: v[0] + v[1]), 2, real_line)
    check("mul",                static(lambda v: v[0] * v[1]), 2, real_line)
    check("sub",                static(lambda v: v[0] - v[1]), 2, real_line)
    check("pow (integer)",      static(lambda v: v[0] ** 3), 1, real_line)
    check("pow (fractional)",   static(lambda v: v[0] ** 0.5), 1, positive)
    check("div",                static(lambda v: v[0] / v[1]), 2, positive)
    check("tanh",               static(lambda v: v[0].tanh()), 1, real_line)
    check("relu",               static(lambda v: v[0].relu()), 1, off_the_kink)
    check("exp",                static(lambda v: v[0].exp()), 1, real_line)
    check("log",                static(lambda v: v[0].log()), 1, positive)
    check("reused input (a*a)", static(lambda v: v[0] * v[0]), 1, real_line)
    check("random graphs",      lambda: make_random_graph(6), 6, real_line)
    print("\nall gradients verified against numerical differentiation")


if __name__ == "__main__":
    main()
