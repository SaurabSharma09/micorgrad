"""
nn.py — neural network building blocks

Builds on top of engine.py to create:
    - Neuron:    single artificial neuron (weights + bias + activation)
    - Layer:     group of neurons all receiving the same input
    - MLP:       multiple layers stacked = full neural network
    - Dropout:   randomly zeros neurons during training (prevents overfitting)
    - BatchNorm: normalizes inputs per batch (stabilizes training)

This is exactly how PyTorch's nn.Module works internally.
Each class has parameters() which returns all learnable weights and biases
as a flat list so the optimizer can update them all in one loop.

Architecture diagram:
    Input → [Layer1] → [Dropout] → [Layer2] → [BatchNorm] → [Layer3] → Output
              MLP handles stacking automatically
"""

import random
import math
from engine import Value


class Module:
    """
    Base class for all neural network components.

    Every component (Neuron, Layer, MLP, Dropout, BatchNorm)
    inherits from this so they all share:
        - parameters(): returns all learnable weights and biases
        - zero_grad():  resets all gradients to 0.0

    This mirrors PyTorch's nn.Module base class exactly.
    """

    def parameters(self):
        """
        Returns all learnable parameters as a flat list of Value objects.
        Override this in each subclass.
        """
        return []

    def zero_grad(self):
        """
        Resets all gradients to 0.0.

        MUST be called before each training iteration.
        Without this, gradients accumulate across iterations because
        _backward uses += not =.

        Example:
            for step in range(100):
                loss = compute_loss(model, xs, ys)
                loss.backward()
                optimizer.step()
                model.zero_grad()   # reset before next iteration
        """
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):
    """
    A single artificial neuron.

    Computes: output = activation(x1*w1 + x2*w2 + ... + xn*wn + b)

    Each neuron has:
        - nin weights (one per input)
        - 1 bias
        - 1 activation function

    The weights and bias are initialized randomly in (-1, 1).
    They will be learned during training via gradient descent.

    Example:
        n = Neuron(nin=3, activation='tanh')
        x = [Value(2.0), Value(-1.0), Value(0.5)]
        out = n(x)   # single Value output
    """

    def __init__(self, nin, activation='tanh'):
        """
        Args:
            nin:        number of inputs this neuron receives
                        determines how many weights are created
            activation: which activation function to use
                        'tanh'    → output in (-1, +1), good for hidden layers
                        'relu'    → output in [0, +inf), most common in deep nets
                        'sigmoid' → output in (0, 1), good for output layer
                        'linear'  → no activation, output = raw weighted sum
        """
        # one weight per input, initialized randomly in (-1, 1)
        # random initialization breaks symmetry so neurons learn different things
        # if all weights were the same, all neurons would learn identically
        self.w = [Value(random.uniform(-1, 1), label=f'w{i}') for i in range(nin)]

        # one bias per neuron, also randomly initialized
        # bias lets the neuron fire even when all inputs are zero
        self.b = Value(random.uniform(-1, 1), label='b')

        # store which activation to use
        self.activation = activation

    def __call__(self, x):
        """
        Forward pass: compute the neuron's output given inputs x.

        Args:
            x: list of inputs (Value objects or plain floats)

        Returns:
            single Value representing the neuron's output

        Steps:
            1. zip pairs each input with its weight: [(w1,x1), (w2,x2), ...]
            2. multiply each pair: [w1*x1, w2*x2, ...]
            3. sum everything + bias: w1*x1 + w2*x2 + ... + b  (cell body)
            4. apply activation function: tanh(cell body)
        """
        # cell body: weighted sum of inputs + bias
        # zip(self.w, x) pairs weights with inputs: [(w1,x1), (w2,x2), ...]
        # wi*xi multiplies each pair
        # sum(...) + self.b adds them all together
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        # apply activation function to squash the output
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        elif self.activation == 'linear':
            # no activation, return raw weighted sum
            # used for regression output or when you want unbounded output
            return act
        else:
            raise ValueError(f"Unknown activation: {self.activation}. "
                             f"Choose from: tanh, relu, sigmoid, linear")

    def parameters(self):
        """
        Returns all learnable parameters: weights + bias.

        Example for Neuron(3):
            returns [w0, w1, w2, b]  → 4 parameters total
        """
        # self.w is already a list
        # [self.b] wraps bias in a list so we can concatenate
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(nin={len(self.w)}, activation={self.activation})"


class Layer(Module):
    """
    A layer of neurons all receiving the same input.

    Every neuron in the layer:
        - receives ALL inputs
        - has its own independent weights
        - produces one output

    So a Layer(nin=3, nout=4) produces 4 outputs from 3 inputs.

    Example:
        L = Layer(nin=3, nout=4, activation='relu')
        x = [Value(2.0), Value(-1.0), Value(0.5)]
        outs = L(x)   # list of 4 Values
    """

    def __init__(self, nin, nout, activation='tanh'):
        """
        Args:
            nin:        number of inputs each neuron receives
            nout:       number of neurons in this layer = number of outputs
            activation: activation function for all neurons in this layer
        """
        # create nout neurons each expecting nin inputs
        # each neuron has its own random weights so they learn different patterns
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        """
        Forward pass: run all neurons on the same input x.

        Every neuron gets the exact same input x.
        Each produces a different output because they have different weights.

        Returns:
            list of nout Values (one per neuron)
            if only 1 neuron, returns just the Value (not a list)
        """
        outs = [n(x) for n in self.neurons]

        # if only one output neuron, return the Value directly
        # this is convenient for the output layer of a classifier
        # loss functions and training loops expect a single Value not a list
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        """
        Returns all parameters from all neurons as one flat list.

        Example for Layer(3, 2):
            neuron1: [w0, w1, w2, b]
            neuron2: [w0, w1, w2, b]
            returns: [w0, w1, w2, b, w0, w1, w2, b]  → 8 parameters total
        """
        # nested list comprehension:
        # for each neuron n in self.neurons
        # for each parameter p in n.parameters()
        # collect p
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer(neurons={len(self.neurons)}, activation={self.neurons[0].activation})"


class MLP(Module):
    """
    Multi-Layer Perceptron: multiple layers stacked together.

    Output of each layer feeds directly into the next layer.
    This is the fundamental building block of all deep learning.

    The universal approximation theorem says:
    an MLP with enough neurons can approximate ANY function.
    This is why it works for images, text, speech — everything.

    Example:
        model = MLP(nin=2, layer_sizes=[16, 16, 1])
        # layer1: Layer(2,  16)  →  16 outputs
        # layer2: Layer(16, 16)  →  16 outputs
        # layer3: Layer(16, 1)   →  1 output (final prediction)

        x = [Value(1.0), Value(-2.0)]
        out = model(x)   # single Value output
    """

    def __init__(self, nin, layer_sizes, activation='tanh', output_activation='linear'):
        """
        Args:
            nin:               number of input features
            layer_sizes:       list of neuron counts per layer
                               e.g. [16, 16, 1] = two hidden layers + output
            activation:        activation for all hidden layers
            output_activation: activation for the final output layer
                               'linear'  → regression (unbounded output)
                               'sigmoid' → binary classification (0 to 1)
                               'tanh'    → output in (-1, +1)
        """
        # sz combines input size with layer sizes
        # e.g. nin=2, layer_sizes=[16,16,1] → sz=[2, 16, 16, 1]
        # this lets us pair up input/output sizes for each layer:
        #   Layer(sz[0], sz[1]) = Layer(2,  16)
        #   Layer(sz[1], sz[2]) = Layer(16, 16)
        #   Layer(sz[2], sz[3]) = Layer(16, 1)
        sz = [nin] + layer_sizes

        # create all layers except the last one with hidden activation
        # create the last layer with output activation
        self.layers = []
        for i in range(len(layer_sizes)):
            is_last = (i == len(layer_sizes) - 1)
            act = output_activation if is_last else activation
            self.layers.append(Layer(sz[i], sz[i + 1], act))

    def __call__(self, x):
        """
        Forward pass: run input through every layer sequentially.

        Output of layer1 becomes input of layer2.
        Output of layer2 becomes input of layer3.
        And so on until the final output.

        Args:
            x: list of input Values or plain floats

        Returns:
            final output Value (or list of Values for multi-output)
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Returns ALL parameters from ALL layers as one flat list.

        Example for MLP(2, [16, 16, 1]):
            layer1: Layer(2,  16) → 2*16 + 16 = 48 parameters
            layer2: Layer(16, 16) → 16*16 + 16 = 272 parameters
            layer3: Layer(16, 1)  → 16*1  + 1  = 17 parameters
            total: 337 parameters
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP(\n" + "\n".join(f"  {l}" for l in self.layers) + "\n)"


class Dropout(Module):
    """
    Dropout layer: randomly zeros neurons during training.

    WHY IT EXISTS:
    During training, neurons can become co-dependent.
    One neuron learns to compensate for mistakes of another.
    This is called co-adaptation and leads to overfitting.

    Dropout forces neurons to be independently useful by randomly
    turning off a fraction of them during each forward pass.
    This makes the network more robust.

    During inference (testing): all neurons are active,
    but outputs are scaled by (1 - p) to maintain the same expected value.

    This is used in every modern transformer including GPT and BERT.

    Example:
        drop = Dropout(p=0.5)
        x = [Value(1.0), Value(2.0), Value(3.0), Value(4.0)]
        out = drop(x, training=True)
        # roughly half the values become 0.0
    """

    def __init__(self, p=0.5):
        """
        Args:
            p: probability of DROPPING a neuron (setting it to 0)
               p=0.5 means 50% of neurons are zeroed randomly
               p=0.0 means no dropout (pass through unchanged)
               p=1.0 means zero everything (useless, avoid)

               typical values: 0.1 to 0.5
               transformers often use p=0.1
        """
        assert 0 <= p < 1, "dropout probability must be in [0, 1)"
        self.p = p

    def __call__(self, x, training=True):
        """
        Forward pass: apply dropout mask to inputs.

        We use INVERTED DROPOUT:
            - training:  randomly zero neurons, scale survivors by 1/(1-p)
            - inference: pass through UNCHANGED (no scaling needed)

        Why inverted dropout?
            By scaling during training, the expected value of each neuron
            stays the same whether dropout is on or off. This means at
            inference time we don't need to touch anything — the activations
            are already correctly scaled.

            Standard dropout would require scaling by (1-p) at inference,
            which is error-prone and easy to forget.

        Args:
            x:        list of Value objects
            training: if True, apply dropout randomly
                      if False, pass through unchanged (inverted dropout)

        Returns:
            list of Value objects (some zeroed if training=True)
        """
        if not training or self.p == 0.0:
            # inverted dropout: inference is a simple pass-through
            # no scaling needed because training already compensated
            return list(x)

        # training mode: randomly zero neurons with probability p
        # each neuron independently has probability p of being dropped
        out = []
        for xi in x:
            if random.random() < self.p:
                # drop this neuron: multiply by 0
                out.append(xi * 0.0)
            else:
                # keep this neuron: scale up by 1/(1-p)
                # this is called inverted dropout
                # scaling during training (not inference) keeps expected value same
                # so we do not need to change anything at inference time
                out.append(xi * (1.0 / (1.0 - self.p)))

        return out

    def parameters(self):
        # dropout has no learnable parameters
        # it only has p which is a hyperparameter set by you, not learned
        return []

    def __repr__(self):
        return f"Dropout(p={self.p})"


class BatchNorm(Module):
    """
    Batch Normalization: normalizes inputs across a batch.

    WHY IT EXISTS:
    As data flows through deep networks, the distribution of each layer's
    inputs keeps shifting because the weights of previous layers change.
    This is called internal covariate shift.

    BatchNorm fixes this by normalizing each feature to have:
        mean = 0
        variance = 1

    Then it learns two parameters:
        gamma (scale): learned scale factor
        beta  (shift): learned shift factor

    Benefits:
        - allows higher learning rates
        - reduces sensitivity to weight initialization
        - acts as a slight regularizer (like dropout)
        - used in ResNets, transformers, and almost every modern network

    Example:
        bn = BatchNorm(nin=4)
        x = [Value(1.0), Value(5.0), Value(2.0), Value(8.0)]
        out = bn(x)
        # outputs are normalized to mean~0, std~1 then rescaled by gamma+beta
    """

    def __init__(self, nin, eps=1e-5):
        """
        Args:
            nin: number of features (inputs) to normalize
            eps: small constant added to variance for numerical stability
                 prevents division by zero when variance is very small
        """
        self.nin = nin
        self.eps = eps

        # gamma: learnable scale parameter, initialized to 1
        # the network learns how much to scale the normalized values
        self.gamma = [Value(1.0, label=f'gamma{i}') for i in range(nin)]

        # beta: learnable shift parameter, initialized to 0
        # the network learns how much to shift the normalized values
        self.beta = [Value(0.0, label=f'beta{i}')  for i in range(nin)]

    def __call__(self, x):
        """
        Forward pass: normalize inputs then rescale with gamma and beta.

        IMPORTANT: every operation uses Value arithmetic so the computation
        stays in the autograd graph. This is critical — if we used plain
        floats (.data), gradients could not flow backward through BatchNorm.

        Steps:
            1. compute mean of inputs          (Value operation)
            2. compute variance of inputs      (Value operation)
            3. normalize: (x - mean) / sqrt(variance + eps)
            4. rescale:   gamma * normalized + beta

        Args:
            x: list of Value objects (length must equal nin)

        Returns:
            list of normalized Value objects (same length)
        """
        assert len(x) == self.nin, (
            f"BatchNorm expected {self.nin} inputs, got {len(x)}"
        )

        n = len(x)

        # step 1: compute mean using Value operations (stays in graph)
        # sum(x) uses Value.__add__ so it builds nodes in the computation graph
        mean = sum(x, Value(0.0)) * Value(1.0 / n)

        # step 2: compute variance using Value operations
        # variance = (1/n) * sum( (xi - mean)^2 )
        diffs_sq = [(xi - mean) ** 2 for xi in x]
        var = sum(diffs_sq, Value(0.0)) * Value(1.0 / n)

        # step 3: normalize each input
        # subtract mean, divide by standard deviation
        # eps prevents division by zero when variance is very small
        # sqrt via ** 0.5 (Value.__pow__ supports float exponents)
        std = (var + Value(self.eps)) ** 0.5
        x_norm = [(xi - mean) / std for xi in x]

        # step 4: rescale with learned gamma and beta
        # gamma and beta let the network undo the normalization if needed
        # if gamma=1 and beta=0: output = normalized input (pure normalization)
        # if gamma=2 and beta=1: output = 2 * normalized + 1 (learned rescaling)
        out = [self.gamma[i] * x_norm[i] + self.beta[i] for i in range(n)]

        return out

    def parameters(self):
        """
        Returns gamma and beta as learnable parameters.
        These are updated during training just like weights and biases.
        """
        return self.gamma + self.beta

    def __repr__(self):
        return f"BatchNorm(nin={self.nin})"
