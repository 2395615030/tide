"""seed1: 记录 attn 和 gated 的 val 曲线, 看交叉点。"""
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
T, B, STEPS = 96, 32, 500
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

def run(kind, seed=1):
    np.random.seed(seed)
    model = (TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu")
             if kind == "attn" else StateLM(V, d=128, n_layers=3, max_T=T, gated=True))
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(seed * 13 + 5)
    curve = []
    t0 = time.time()
    for s in range(STEPS):
        x, y = get(tr, B, rng)
        out = model(Tensor(x), mask) if kind == "attn" else model(Tensor(x))
        loss = cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0 or s == STEPS - 1:
            v = evaluate(model, kind)
            curve.append((s, v))
            print(f"  [{kind:5s}] step {s:3d} train {float(loss.data):.3f} val {v:.4f} "
                  f"{(time.time()-t0)/(s+1)*1000:.0f}ms", flush=True)
    return curve

ca = run("attn")
cg = run("gated")
print("\n=== val curve comparison (seed 1) ===")
print("step  attn    gated   gap")
for (s, a), (_, g) in zip(ca, cg):
    print(f"{s:4d}  {a:.4f}  {g:.4f}  {(g-a)/a*100:+6.2f}%")
