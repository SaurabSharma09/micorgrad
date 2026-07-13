"""
optim.py - Optimizers for micrograd-explained
==============================================
Implements SGD and Adam from scratch using only the Value class.

Why optimizers matter:
  After loss.backward() fills every parameter's .grad,
  the optimizer's job is to update .data using that gradient.
  Different optimizers do this in smarter ways than plain SGD.

Two optimizers here:
  SGD  → simplest possible: move opposite to gradient
  Adam → what every real LLM uses: adaptive learning rates
         with momentum + variance tracking
"""


class SGD:
    """
    Stochastic Gradient Descent.

    Update rule (for each parameter p):
        p.data -= lr * p.grad

    That's it. Move the parameter a small step
    in the direction that reduces loss.

    Why 'stochastic'?
        In practice we compute gradients on a random mini-batch,
        not the whole dataset. That randomness is the 'stochastic' part.

    Args:
        parameters : list of Value objects (from model.parameters())
        lr         : learning rate, controls step size (default 0.01)
        weight_decay: L2 regularization strength (default 0.0)
                      adds lr * wd * p.data to the update,
                      which shrinks weights toward zero each step
    """

    def __init__(self, parameters, lr=0.01, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        """
        Apply one gradient update to every parameter.
        Call this AFTER loss.backward().
        """
        for p in self.parameters:
            if p.grad is None:
                continue  # skip params that have no gradient yet

            # L2 weight decay: effectively adds a penalty for large weights
            # grad becomes: p.grad + weight_decay * p.data
            grad = p.grad + self.weight_decay * p.data

            # core SGD update: move opposite to gradient
            p.data -= self.lr * grad

    def zero_grad(self):
        """
        Reset all gradients to zero before the next forward pass.
        MUST call this before loss.backward() each step,
        otherwise gradients accumulate across steps.
        """
        for p in self.parameters:
            p.grad = 0.0

    def __repr__(self):
        return f"SGD(lr={self.lr}, weight_decay={self.weight_decay})"


class Adam:
    """
    Adam: Adaptive Moment Estimation.
    Paper: Kingma & Ba, 2014 (https://arxiv.org/abs/1412.6980)

    Why Adam is better than SGD:
        SGD uses the same learning rate for every parameter.
        Adam tracks a moving average of the gradient (momentum)
        AND a moving average of the squared gradient (variance).
        Parameters that receive large, consistent gradients get
        smaller effective learning rates. Sparse gradients get larger.
        This is why LLMs use Adam: embedding rows that are rarely
        updated still get meaningful updates when they are.

    Update rule (for each parameter p at timestep t):

        m_t = beta1 * m_{t-1} + (1 - beta1) * grad        # 1st moment (mean)
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2      # 2nd moment (variance)

        m_hat = m_t / (1 - beta1^t)    # bias correction for 1st moment
        v_hat = v_t / (1 - beta2^t)    # bias correction for 2nd moment

        p.data -= lr * m_hat / (sqrt(v_hat) + eps)

    Bias correction explained:
        At t=1, m_1 = (1-beta1)*grad ≈ 0.1*grad (if beta1=0.9).
        This underestimates the true gradient.
        Dividing by (1 - beta1^1) = 0.1 corrects for this cold start.
        As t grows large, beta1^t → 0 and the correction vanishes.

    Args:
        parameters : list of Value objects
        lr         : step size (default 0.001, much smaller than SGD)
        beta1      : decay for 1st moment (default 0.9)
        beta2      : decay for 2nd moment (default 0.999)
        eps        : small constant for numerical stability (default 1e-8)
                     prevents division by zero when variance is tiny
        weight_decay: L2 regularization (default 0.0)
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        # timestep counter — used for bias correction
        self.t = 0

        # first moment (mean of gradients) — initialized to 0 for each param
        # key: id(p) so we can look up the state for each Value object
        self.m = {id(p): 0.0 for p in self.parameters}

        # second moment (mean of squared gradients) — also starts at 0
        self.v = {id(p): 0.0 for p in self.parameters}

    def step(self):
        """
        Apply one Adam update to every parameter.
        Call this AFTER loss.backward().
        """
        self.t += 1  # increment global step counter

        for p in self.parameters:
            if p.grad is None:
                continue

            # optional: L2 regularization baked into gradient
            grad = p.grad + self.weight_decay * p.data

            pid = id(p)  # use object id as dict key

            # --- 1st moment update ---
            # exponential moving average of gradient
            # beta1=0.9 means: 90% old momentum, 10% new gradient
            self.m[pid] = self.beta1 * self.m[pid] + (1.0 - self.beta1) * grad

            # --- 2nd moment update ---
            # exponential moving average of squared gradient
            # tracks how noisy / large the gradient has been
            self.v[pid] = self.beta2 * self.v[pid] + (1.0 - self.beta2) * (grad ** 2)

            # --- bias correction ---
            # early in training, m and v are biased toward 0
            # dividing corrects for this initialization effect
            m_hat = self.m[pid] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[pid] / (1.0 - self.beta2 ** self.t)

            # --- parameter update ---
            # divide by sqrt(v_hat) + eps:
            #   large variance → smaller step (we're uncertain)
            #   small variance → larger step (gradient is consistent)
            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)

    def zero_grad(self):
        """
        Reset all gradients to zero before the next forward pass.
        Does NOT reset m and v — those persist across steps by design.
        """
        for p in self.parameters:
            p.grad = 0.0

    def __repr__(self):
        return (f"Adam(lr={self.lr}, beta1={self.beta1}, "
                f"beta2={self.beta2}, eps={self.eps})")
