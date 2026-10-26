"""1000 步长跑: attention vs 门控状态, 看差距是否随训练扩大。"""
import sys, os, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
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
T, B, STEPS = 96, 32, 1000
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)

def get(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    return (np.stack([arr[i:i+T] for i in ix]),
            np.stack([arr[i+1:i+T+1] for i in ix]))

def evaluate(model, kind, n=10):
    rng = np.random.RandomState(12345)   # 固定验证集
    tot = 0.0
    for _ in range(n):
        x, y = get(va, B, rng)
        out = model(Tensor(x), mask) if kind == "attn" else model(Tensor(x))
        tot += float(cross_entropy(out, y).data)
    return tot / n

def make(kind):
    np.random.seed(42)
    if kind == "attn":
        return TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu")
    return StateLM(V, d=128, n_layers=3, max_T=T, gated=True)

models, opts = {}, {}
for k in ("attn", "gated"):
    models[k] = make(k)
    opts[k] = AdamW(models[k].params(), lr=3e-3, weight_decay=0.01)
rngs = {"attn": np.random.RandomState(0), "gated": np.random.RandomState(0)}

t0 = time.time()
hist = {"attn": [], "gated": []}
for s in range(STEPS):
    for k in ("attn", "gated"):
        x, y = get(tr, B, rngs[k])
        model = models[k]
        out = model(Tensor(x), mask) if k == "attn" else model(Tensor(x))
        loss = cross_entropy(out, y)
        opts[k].zero_grad(); loss.backward(); opts[k].step()
    if s % 100 == 0 or s == STEPS - 1:
        a = evaluate(models["attn"], "attn")
        g = evaluate(models["gated"], "gated")
        hist["attn"].append((s, a))
        hist["gated"].append((s, g))
        print(f"step {s:4d}  attn {a:.4f}  gated {g:.4f}  gap {(g-a)/a*100:+.2f}%  "
              f"{(time.time()-t0):.0f}s", flush=True)

print("\n=== GAP TREND ===")
for (s, a), (_, g) in zip(hist["attn"], hist["gated"]):
    print(f"step {s:4d}  gap {(g-a)/a*100:+.2f}%")
print(f"total {time.time()-t0:.0f}s")
