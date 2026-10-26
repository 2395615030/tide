"""公平对比: attn vs gated, 都用 wd=0.3, 3 seed, 500 步, 取最佳 val。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.statelm import StateLM
from nano_torch.nn.transformer import TinyLM
from nano_torch.data.neko import load_neko

text, ids, keep = load_neko()
data = np.array(ids, dtype=np.int64)
split = int(len(data) * 0.9)
tr, va = data[:split], data[split:]
V = len(keep) + 1
T, B, STEPS, WD = 96, 32, 500, 0.3
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)

def get(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    return (np.stack([arr[i:i+T] for i in ix]),
            np.stack([arr[i+1:i+T+1] for i in ix]))

def evaluate(model, kind):
    vrng = np.random.RandomState(12345)
    tot = 0.0
    for _ in range(6):
        x, y = get(va, B, vrng)
        out = model(Tensor(x), mask) if kind == "attn" else model(Tensor(x))
        tot += float(cross_entropy(out, y).data)
    return tot / 6

def run(kind, seed):
    np.random.seed(seed)
    model = (TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu")
             if kind == "attn" else StateLM(V, d=128, n_layers=3, max_T=T, gated=True))
    opt = AdamW(model.params(), lr=3e-3, weight_decay=WD)
    rng = np.random.RandomState(seed * 13 + 5)
    best = (9e9, 0)
    for s in range(STEPS):
        x, y = get(tr, B, rng)
        out = model(Tensor(x), mask) if kind == "attn" else model(Tensor(x))
        loss = cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0 or s == STEPS - 1:
            v = evaluate(model, kind)
            if v < best[0]: best = (v, s)
    return best

t0 = time.time()
deltas = []
for seed in (0, 1, 2):
    a = run("attn", seed)
    g = run("gated", seed)
    d = (g[0] - a[0]) / a[0] * 100
    deltas.append(d)
    print(f"seed {seed}: attn {a[0]:.4f}@{a[1]}  gated {g[0]:.4f}@{g[1]}  "
          f"delta {d:+.2f}%  [{time.time()-t0:.0f}s]", flush=True)
print(f"\nmean delta {np.mean(deltas):+.2f}%  std {np.std(deltas):.2f}%")
