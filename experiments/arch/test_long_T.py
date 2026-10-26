"""长序列测试: T=384, 看状态模型 vs attention 的差距变化。"""
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
T, B, STEPS = 384, 16, 200
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
print(f"T={T} B={B} vocab={V}", flush=True)

def get(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    return (np.stack([arr[i:i+T] for i in ix]),
            np.stack([arr[i+1:i+T+1] for i in ix]))

def evaluate(model, kind, n=4):
    rng = np.random.RandomState(777)
    tot = 0.0
    for _ in range(n):
        x, y = get(va, B, rng)
        out = model(Tensor(x), mask) if kind == "attn" else model(Tensor(x))
        tot += float(cross_entropy(out, y).data)
    return tot / n

def run(kind):
    np.random.seed(42)
    model = (TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu")
             if kind == "attn" else StateLM(V, d=128, n_layers=3, max_T=T, gated=True))
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(0)
    t0 = time.time()
    for s in range(STEPS):
        x, y = get(tr, B, rng)
        out = model(Tensor(x), mask) if kind == "attn" else model(Tensor(x))
        loss = cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 50 == 0 or s == STEPS - 1:
            print(f"  [{kind:5s}] {s:3d} train {float(loss.data):.3f}  val {evaluate(model, kind):.3f}  "
                  f"{(time.time()-t0)/(s+1)*1000:.0f} ms", flush=True)
    return evaluate(model, kind, n=6), time.time() - t0

va1, ta = run("attn")
vg1, tg = run("gated")
print(f"\nT=384  attn val {va1:.4f}  {ta:.0f}s ({ta/STEPS*1000:.0f} ms/step)")
print(f"T=384  gated val {vg1:.4f}  {tg:.0f}s ({tg/STEPS*1000:.0f} ms/step)")
print(f"gap {(vg1-va1)/va1*100:+.2f}%")
