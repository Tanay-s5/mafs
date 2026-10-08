"""The maths behind the demo.  Plain NumPy, nothing else.

x is the 28x28 drawing flattened into a vector of 784 numbers in [0, 1].

Scene 1  (no hidden layers)      y  = softmax(W x + b)            W : 10 x 784
Scene 2  (two hidden layers)     a1 = sigmoid(W1 x  + b1)         W1: 32 x 784
                                 a2 = sigmoid(W2 a1 + b2)         W2: 16 x 32
                                 y  = softmax(W3 a2 + b3)         W3: 10 x 16

Every layer is one matrix-vector product, a shift (bias) and a squashing function.
"""
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).with_name("weights.npz")
H1, H2 = 32, 16                      # hidden layer sizes used in Scene 2


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def load_weights(path=WEIGHTS):
    w = np.load(path)
    return {k: w[k].astype(np.float64) for k in w.files}


# ---------------------------------------------------------------- forward passes
def linear_scores(w, x):
    """Scene 1 raw scores: x (784,) -> z (10,)."""
    return w["L_W"] @ x + w["L_b"]


def linear_forward(w, x):
    """Scene 1 probabilities: x (784,) -> y (10,)."""
    return softmax(linear_scores(w, x))


def hidden_forward(w, x):
    """Scene 2.  x: (784,)  ->  (a1: (32,), a2: (16,), y: (10,))."""
    a1 = sigmoid(w["W1"] @ x + w["b1"])
    a2 = sigmoid(w["W2"] @ a1 + w["b2"])
    y = softmax(w["W3"] @ a2 + w["b3"])
    return a1, a2, y


# ---------------------------------------------------------------- centring
def centre_shift(img):
    """Whole-pixel (dy, dx) that moves the ink's centre of mass to the middle of
    the 28x28 grid.  MNIST digits are centred this way, so the models expect it."""
    total = img.sum()
    if total < 1e-6:
        return 0, 0
    ys, xs = np.mgrid[0:28, 0:28]
    cy = (img * (ys + 0.5)).sum() / total
    cx = (img * (xs + 0.5)).sum() / total
    dy, dx = int(round(14.0 - cy)), int(round(14.0 - cx))
    rows = np.where(img.max(axis=1) > 0.05)[0]          # never push ink off the grid
    cols = np.where(img.max(axis=0) > 0.05)[0]
    dy = int(np.clip(dy, -rows.min(), 27 - rows.max()))
    dx = int(np.clip(dx, -cols.min(), 27 - cols.max()))
    return dy, dx


def shift_image(img, dy, dx):
    """Slide the image by whole pixels (dy down, dx right), filling with zeros."""
    out = np.zeros_like(img)
    h, w = img.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        img[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out