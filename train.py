"""Train the two models used in the presentation.  NumPy only (no PyTorch / TensorFlow).

    python train.py                       # downloads MNIST once (~11 MB), trains, writes weights.npz
    python train.py --mnist my_mnist.npz  # use a local keras-style npz instead of downloading
    python train.py --only hidden         # retrain just one model ("linear" or "hidden")

Takes a couple of minutes on a laptop CPU.  Run it once before the presentation.

Both models are trained on randomly rotated / scaled / sheared / shifted / thickened copies
of MNIST, so digits drawn straight onto a 28x28 grid (by mouse) are recognised too.
"""
import argparse
import gzip
import io
import pickle
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np

from network import H1, H2, WEIGHTS, sigmoid, softmax

CACHE = Path(__file__).resolve().parent / "mnist_cache" / "mnist.npz"

IDX_FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
IDX_MIRRORS = ("https://ossci-datasets.s3.amazonaws.com/mnist/",
               "https://storage.googleapis.com/cvdf-datasets/mnist/")
KERAS_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
PICKLE_URL = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"


# ------------------------------------------------------------------ data
def download(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def from_idx(base):
    raw = [gzip.decompress(download(base + f)) for f in IDX_FILES]
    imgs = lambda b: np.frombuffer(b, np.uint8, offset=16).reshape(-1, 28, 28)
    labs = lambda b: np.frombuffer(b, np.uint8, offset=8)
    return imgs(raw[0]), labs(raw[1]), imgs(raw[2]), labs(raw[3])


def from_keras():
    d = np.load(io.BytesIO(download(KERAS_URL)))
    return d["x_train"], d["y_train"], d["x_test"], d["y_test"]


def from_pickle():
    with warnings.catch_warnings():                       # old numpy pickle -> harmless warning
        warnings.simplefilter("ignore")
        tr, va, te = pickle.loads(gzip.decompress(download(PICKLE_URL)), encoding="latin1")
    to_u8 = lambda a: (a.reshape(-1, 28, 28) * 255).round().astype(np.uint8)
    return (to_u8(np.concatenate([tr[0], va[0]])), np.concatenate([tr[1], va[1]]),
            to_u8(te[0]), te[1])


def load_mnist(local=None):
    if local:
        d = np.load(local)
        arrays = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    elif CACHE.exists():
        d = np.load(CACHE)
        arrays = d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    else:
        arrays = None
        attempts = [lambda b=b: from_idx(b) for b in IDX_MIRRORS] + [from_keras, from_pickle]
        for get in attempts:
            try:
                print("downloading MNIST ...")
                arrays = get()
                break
            except Exception as e:                                   # try the next mirror
                print("  failed (%s), trying another mirror" % type(e).__name__)
        if arrays is None:
            raise SystemExit("Could not download MNIST. Download mnist.npz manually "
                             "and run:  python train.py --mnist mnist.npz")
        CACHE.parent.mkdir(exist_ok=True)
        np.savez_compressed(CACHE, x_train=arrays[0], y_train=arrays[1],
                            x_test=arrays[2], y_test=arrays[3])
    xtr, ytr, xte, yte = arrays
    return (xtr.astype(np.float32) / 255, ytr.astype(np.int64),
            xte.astype(np.float32) / 255, yte.astype(np.int64))


# ------------------------------------------------------------------ augmentation
def random_warp(x, rng, rot, scale, shear, shift):
    """Randomly rotate / scale / shear / shift each 28x28 image (bilinear, NumPy only)."""
    n = len(x)
    g = np.stack(np.meshgrid(np.arange(28) - 13.5, np.arange(28) - 13.5), -1)
    g = g.reshape(-1, 2).astype(np.float32)                       # (784, 2) as (x, y)
    a = np.deg2rad(rng.uniform(-rot, rot, n))
    sc = rng.uniform(scale[0], scale[1], n)
    sh = rng.uniform(-shear, shear, n)
    c, s = np.cos(a), np.sin(a)
    M = sc[:, None, None] * np.stack([np.stack([c, c * sh - s], -1),
                                      np.stack([s, s * sh + c], -1)], 1)      # (n, 2, 2)
    Minv = np.linalg.inv(M)
    t = rng.uniform(-shift, shift, (n, 2))
    src = (np.einsum("nij,pj->npi", Minv, g)
           - np.einsum("nij,nj->ni", Minv, t)[:, None, :] + 13.5).astype(np.float32)
    sx, sy = src[..., 0], src[..., 1]
    x0, y0 = np.floor(sx).astype(np.int32), np.floor(sy).astype(np.int32)
    fx, fy = sx - x0, sy - y0
    flat = x.reshape(n, 784)

    def at(yy, xx):
        ok = (xx >= 0) & (xx < 28) & (yy >= 0) & (yy < 28)
        return np.take_along_axis(flat, np.clip(yy, 0, 27) * 28 + np.clip(xx, 0, 27), axis=1) * ok

    out = (at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x0 + 1) * fx * (1 - fy)
           + at(y0 + 1, x0) * (1 - fx) * fy + at(y0 + 1, x0 + 1) * fx * fy)
    return out.reshape(n, 28, 28)


def vary_thickness(x, rng):
    """Thicken some strokes, soften others (a mouse brush is not exactly a pen)."""
    p = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    up, down, left, right = p[:, :-2, 1:-1], p[:, 2:, 1:-1], p[:, 1:-1, :-2], p[:, 1:-1, 2:]
    r = rng.random(len(x))
    thick = np.maximum(x, 0.8 * np.maximum.reduce([up, down, left, right]))
    soft = 0.5 * x + 0.125 * (up + down + left + right)
    out = x.copy()
    m = r < 0.35
    out[m] = thick[m]
    m = (r >= 0.35) & (r < 0.55)
    out[m] = soft[m]
    return np.clip(out, 0, 1)


def augment(x, rng, aug, chunk=5000):
    out = np.empty_like(x)
    for s in range(0, len(x), chunk):
        out[s:s + chunk] = vary_thickness(random_warp(x[s:s + chunk], rng, **aug), rng)
    return out


# ------------------------------------------------------------------ models
def init_layer(rng, n_out, n_in):
    return rng.normal(0, np.sqrt(1.0 / n_in), (n_out, n_in)), np.zeros(n_out)


def lin_forward(p, X):                                  # X: (batch, 784)
    return softmax(X @ p["L_W"].T + p["L_b"])

def linear_scores(p, x):
    return x @ p["L_W"].T + p["L_b"]


def lin_grads(p, X, Y):
    d = (lin_forward(p, X) - Y) / len(X)
    return {"L_W": d.T @ X, "L_b": d.sum(0)}


def mlp_forward(p, X):
    a1 = sigmoid(X @ p["W1"].T + p["b1"])
    a2 = sigmoid(a1 @ p["W2"].T + p["b2"])
    return a1, a2, softmax(a2 @ p["W3"].T + p["b3"])


def mlp_grads(p, X, Y):                                 # back-propagation = the chain rule in matrix form
    a1, a2, y = mlp_forward(p, X)
    d3 = (y - Y) / len(X)
    d2 = (d3 @ p["W3"]) * a2 * (1 - a2)
    d1 = (d2 @ p["W2"]) * a1 * (1 - a1)
    return {"W3": d3.T @ a2, "b3": d3.sum(0), "W2": d2.T @ a1, "b2": d2.sum(0),
            "W1": d1.T @ X, "b1": d1.sum(0)}


class Adam:
    def __init__(self, params, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.b1, self.b2, self.eps, self.t = params, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads, lr):
        self.t += 1
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + self.eps)


def accuracy(probs_fn, p, X, y):
    return float((probs_fn(p, X).argmax(1) == y).mean())


def fit(name, p, grads_fn, probs_fn, data, aug, epochs, lr0, copies, batch=128, seed=0):
    xtr, ytr, xte, yte = data
    rng = np.random.default_rng(seed)
    opt = Adam(p)
    Y = np.eye(10, dtype=np.float32)[ytr]
    shaken = augment(xte, np.random.default_rng(123), aug).reshape(-1, 784)   # harder test set
    clean = xte.reshape(-1, 784)
    print("%-8s making %d randomly warped copies of the training set ..." % (name, copies), flush=True)
    variants = [(augment(xtr, rng, aug) * 255).round().astype(np.uint8) for _ in range(copies)]
    for ep in range(epochs):
        t0 = time.time()
        X = variants[ep % copies].reshape(len(xtr), 784).astype(np.float32) / 255     # a different copy each epoch
        order = rng.permutation(len(X))
        lr = lr0 * (0.03 + 0.97 * 0.5 * (1 + np.cos(np.pi * ep / epochs)))
        for i in range(0, len(X), batch):
            b = order[i:i + batch]
            opt.step(grads_fn(p, X[b], Y[b]), lr)
        print("%-8s epoch %2d/%d  clean test %.2f%%  shaken test %.2f%%  (%.1fs)" % (
            name, ep + 1, epochs, 100 * accuracy(probs_fn, p, clean, yte),
            100 * accuracy(probs_fn, p, shaken, yte), time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", help="local keras-style mnist.npz (skips the download)")
    ap.add_argument("--only", choices=["linear", "hidden"], help="train just one of the two models")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (quick test)")
    args = ap.parse_args()
    data = load_mnist(args.mnist)
    print("MNIST:", data[0].shape[0], "train /", data[2].shape[0], "test")
    rng = np.random.default_rng(1)
    saved = dict(np.load(WEIGHTS)) if WEIGHTS.exists() else {}

    if args.only != "hidden":
        # Scene 1: no hidden layers  
        p_lin = dict(zip(("L_W", "L_b"), init_layer(rng, 10, 784)))
        fit("linear", p_lin, lin_grads, lambda p, X: lin_forward(p, X), data,
            aug=dict(rot=6, scale=(0.92, 1.08), shear=0.08, shift=1.5),
            epochs=args.epochs or 12, lr0=2e-3, copies=4)
        saved.update(p_lin)
        np.savez(WEIGHTS, **{k: v.astype(np.float32) for k, v in saved.items()})

    if args.only != "linear":
        # Scene 2: 2 hidden layers
        p_mlp = {}
        for i, (n_out, n_in) in enumerate([(H1, 784), (H2, H1), (10, H2)], start=1):
            p_mlp["W%d" % i], p_mlp["b%d" % i] = init_layer(rng, n_out, n_in)
        fit("hidden", p_mlp, mlp_grads, lambda p, X: mlp_forward(p, X)[2], data,
            aug=dict(rot=12, scale=(0.85, 1.15), shear=0.15, shift=2.0),
            epochs=args.epochs or 60, lr0=3e-3, copies=8)
        saved.update(p_mlp)
        np.savez(WEIGHTS, **{k: v.astype(np.float32) for k, v in saved.items()})
    print("saved", WEIGHTS)


if __name__ == "__main__":
    main()