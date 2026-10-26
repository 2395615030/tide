"""三组对比: attention / 纯状态 / 门控状态。同一份 Neko 数据。"""
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
T, B, STEPS = 96, 32, 400
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)

def get(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    return (np.stack([arr[i:i+T] for i in ix]),
            np.stack([arr[i+1:i+T+1] for i in ix]))

def evaluate(model, kind, rng, n=6):
    tot = 0.0
    for _ in range(n):
        x, y = get(va, B, rng)
        out = model(Tensor(x), mask) if kind == "attn" else model(Tensor(x))
        tot += float(cross_entropy(out, y).data)
    return tot / n

def run(kind):
    np.random.seed(42)
    if kind == "attn":
        model = TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu")
    elif kind == "state":
        model = StateLM(V, d=128, n_layers=3, max_T=T, gated=False)
    else:
        model = StateLM(V, d=128, n_layers=3, max_T=T, gated=True)
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(0); vrng = np.random.RandomState(999)
    t0 = time.time()
    nparam = sum(p.size for p in model.params())
    for s in range(STEPS):
        x, y = get(tr, B, rng)
        out = model(Tensor(x), mask) if kind == "attn" else model(Tensor(x))
        loss = cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0 or s == STEPS - 1:
            print(f"  [{kind:6s}] {s:3d} train {float(loss.data):.3f}  "
                  f"val {evaluate(model, kind, vrng):.3f}  {(time.time()-t0)/(s+1)*1000:.0f} ms  "
                  f"params {nparam}", flush=True)
    return evaluate(model, kind, vrng, n=12), time.time() - t0, nparam

res = {}
for kind in ["gated"]:
    res[kind] = run(kind)

print("\n=== RESULT (400 steps) ===")
for k, (v, t, p) in res.items():
    print(f"{k:6s} val {v:.4f}  {t:5.0f}s  params {p}")
base = 3.4294
for k in ["gated"]:
    print(f"{k:6s} vs attn(3.4294): {(res[k][0]-base)/base*100:+.1f}%")
