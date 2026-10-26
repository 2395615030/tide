"""attention 模型 vs 纯状态模型, 同一份 Neko 数据, 同样步数。"""
import sys, os, time, glob, json
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

def evaluate(model, is_state, rng, n=8):
    tot = 0.0
    for _ in range(n):
        x, y = get(va, B, rng)
        out = model(Tensor(x)) if is_state else model(Tensor(x), mask)
        tot += float(cross_entropy(out, y).data)
    return tot / n

def run(kind):
    np.random.seed(42)
    is_state = (kind == "state")
    model = (StateLM(V, d=128, n_layers=3, max_T=T) if is_state
             else TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu"))
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(0); vrng = np.random.RandomState(999)
    t0 = time.time()
    for s in range(STEPS):
        x, y = get(tr, B, rng)
        out = model(Tensor(x)) if is_state else model(Tensor(x), mask)
        loss = cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0 or s == STEPS - 1:
            print(f"  [{kind:5s}] {s:3d} train {float(loss.data):.3f}  val {evaluate(model, is_state, vrng):.3f}  "
                  f"{(time.time()-t0)/(s+1)*1000:.0f} ms", flush=True)
    return evaluate(model, is_state, vrng, n=12), time.time() - t0

v1, t1 = run("attn")
v2, t2 = run("state")
print(f"\nattn  val {v1:.4f}  {t1:.0f}s")
print(f"state val {v2:.4f}  {t2:.0f}s   delta {(v2-v1)/v1*100:+.1f}%")
