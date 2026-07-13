"""
loss.py - Loss functions for micrograd-explained
=================================================
Three loss functions built entirely from Value operations,
so backward() works automatically through them.

Why loss functions matter:
  The loss is a single scalar that measures how wrong the model is.
  backward() on this scalar fills .grad for every parameter.
  The optimizer then uses those grads to improve the model.

Three functions here:
  mse_loss            → regression problems (predict a number)
  binary_cross_entropy → binary classification (yes/no output)
  cross_entropy_loss  → multi-class classification (which of N classes)
"""

import math
from engine import Value


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _to_value(x):
    """Wrap plain floats in Value so loss functions accept either type."""
    return x if isinstance(x, Value) else Value(float(x))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mean Squared Error
# ─────────────────────────────────────────────────────────────────────────────

def mse_loss(predictions, targets):
    """
    Mean Squared Error loss.

    Formula:
        MSE = (1/N) * sum( (pred_i - target_i)^2 )

    When to use:
        Regression — when the output is a continuous number.
        Example: predicting house price, temperature, stock return.

    Why squared?
        (pred - target)^2 is always positive and penalizes large
        errors much more than small ones. A prediction that is 2 off
        gets penalized 4x more than one that is 1 off.

    Args:
        predictions : list of Value objects (model outputs)
        targets     : list of Value or float (ground truth)

    Returns:
        A single Value representing the mean squared error.
        Calling .backward() on it propagates gradients to all params.
    """
    assert len(predictions) == len(targets), (
        f"predictions and targets must have same length, "
        f"got {len(predictions)} and {len(targets)}"
    )

    n = len(predictions)
    # sum of squared differences
    # each (pred - target)**2 builds a node in the computation graph
    total = sum(
        (_to_value(p) - _to_value(t)) ** 2
        for p, t in zip(predictions, targets)
    )
    # divide by N to get the mean
    # using Value(n) so the division is part of the graph
    return total * Value(1.0 / n)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Binary Cross Entropy
# ─────────────────────────────────────────────────────────────────────────────

def binary_cross_entropy(predictions, targets, eps=1e-7):
    """
    Binary Cross Entropy loss (BCE).

    Formula:
        BCE = -(1/N) * sum( t*log(p) + (1-t)*log(1-p) )

    When to use:
        Binary classification — output is a single sigmoid probability.
        Example: spam/not-spam, cat/not-cat, positive/negative sentiment.

    Intuition:
        If true label t=1 and prediction p=0.99:
            loss = -log(0.99) ≈ 0.01   (small, good prediction)
        If true label t=1 and prediction p=0.01:
            loss = -log(0.01) ≈ 4.6    (large, bad prediction)
        The log makes wrong confident predictions very expensive.

    Args:
        predictions : list of Value in range (0, 1) — sigmoid outputs
        targets     : list of Value or float — 0 or 1 labels
        eps         : small clip to avoid log(0) which is -infinity

    Returns:
        A single Value representing the mean BCE loss.
    """
    assert len(predictions) == len(targets)

    n = len(predictions)
    total = Value(0.0)

    for p, t in zip(predictions, targets):
        p = _to_value(p)
        t = _to_value(t)

        # clip prediction to [eps, 1-eps] to prevent log(0)
        # we do this by adding eps (tiny nudge) rather than conditional logic
        # because conditionals can't be part of the computation graph
        #
        # Note: for a pure micrograd implementation we accept that
        # extreme predictions (very close to 0 or 1) may produce
        # large but finite loss values. The eps in log is a safety net.

        # t * log(p + eps)  +  (1-t) * log(1 - p + eps)
        positive_term = t * (p + Value(eps)).log()
        negative_term = (Value(1.0) - t) * (Value(1.0) - p + Value(eps)).log()

        total = total + positive_term + negative_term

    # negate and divide by N
    return Value(-1.0 / n) * total


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cross Entropy Loss (multi-class)
# ─────────────────────────────────────────────────────────────────────────────

def cross_entropy_loss(logits_batch, targets):
    """
    Cross Entropy loss for multi-class classification.

    This is what GPT uses for next-token prediction.
    Each training example has K logits (one per class),
    and the loss pushes the correct class logit up.

    Formula (for one example):
        softmax(z_k) = exp(z_k) / sum_j( exp(z_j) )
        CE = -log( softmax(z_{correct_class}) )
           = -z_{correct} + log( sum_j( exp(z_j) ) )

    Numerical stability trick (log-sum-exp):
        subtract max(z) before exp to prevent overflow.
        This does not change the result because:
            log( sum(exp(z_k - m)) ) + m == log( sum(exp(z_k)) )
        where m = max(z_k).

    Args:
        logits_batch : list of lists — shape (N, K)
                       each inner list has K raw scores (before softmax)
        targets      : list of int — shape (N,)
                       integer class index for each example

    Returns:
        A single Value representing the mean cross entropy loss.
    """
    assert len(logits_batch) == len(targets)

    n = len(logits_batch)
    total = Value(0.0)

    for logits, correct_class in zip(logits_batch, targets):
        logits = [_to_value(z) for z in logits]
        k = len(logits)
        assert 0 <= correct_class < k, (
            f"correct_class {correct_class} out of range [0, {k})"
        )

        # --- numerical stability: subtract max ---
        # plain floats needed for max(), Value objects for computation graph
        max_val = max(z.data for z in logits)

        # exp(z_k - max) for each class
        exps = [(z - Value(max_val)).exp() for z in logits]

        # sum of all exps (denominator of softmax)
        sum_exp = sum(exps[1:], exps[0])  # avoids adding to float 0

        # log of the correct class's softmax probability
        # = (z_correct - max) - log(sum_exp)
        log_prob = (logits[correct_class] - Value(max_val)).log() if False else \
                   (logits[correct_class] - Value(max_val)) - sum_exp.log()

        # accumulate negative log probability
        total = total + (Value(-1.0) * log_prob)

    return total * Value(1.0 / n)
