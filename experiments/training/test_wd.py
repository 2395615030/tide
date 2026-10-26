"""gated 的过拟合能否被 weight decay 压住? seed1, 500 步。"""
import sys, os, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.statelm import StateLM
from nano_torch.data.neko import load_neko

text, ids, keep = load_neko()
data = np.array(ids, dtype=np.int64)
split = int(len(data) * 0.9)
tr, va = data[:split], data[split:]
V = len(keep) + 1
T, B, STEPS = 96, 32, 500

def get(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    return (np.stack([arr[i:i+T] for i in ix]),
            np.stack([arr[i+1:i+T+1] for i in ix]))

def evaluate(model):
    vrng = np.random.RandomState(12345)
    tot = 0.0
    for _ in range(6):
        x, y = get(va, B, vrng)
        tot += float(cross_entropy(model(Tensor(x)), y).data)
    return tot / 6

def run(wd):
    np.random.seed(1)
    model = StateLM(V, d=128, n_layers=3, max_T=T, gated=True)
    opt = AdamW(model.params(), lr=3e-3, weight_decay=wd)
    rng = np.random.RandomState(1 * 13 + 5)
    best = (9e9, 0)
    t0 = time.time()
    for s in range(STEPS):
        x, y = get(tr, B, rng)
        loss = cross_entropy(model(Tensor(x)), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0 or s == STEPS - 1:
            v = evaluate(model)
            if v < best[0]: best = (v, s)
            if s % 100 == 0 or s == STEPS - 1:
                print(f"  wd={wd:<5} step {s:3d} train {float(loss.data):.3f} "
                      f"val {v:.4f} best {best[0]:.4f}@{best[1]}", flush=True)
    return best

for wd in (0.01, 0.1, 0.3):
    b = run(wd)
    print(f"--> wd={wd}: best val {b[0]:.4f} at step {b[1]}\n", flush=True)
