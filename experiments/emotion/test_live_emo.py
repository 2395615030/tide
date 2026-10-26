"""三组: base / fixed / live(相位+振荡)。"""
import sys, os, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.core.live_endo import LiveEndocrine
from nano_torch.nn.statelm import StateLM
from nano_torch.data.neko import load_neko

text, ids, keep = load_neko()
data = np.array(ids, dtype=np.int64)
split = int(len(data) * 0.9)
tr, va = data[:split], data[split:]
V = len(keep) + 1
T, B, STEPS = 96, 32, 300
FIXED = np.array([0.6, 0.0, 0.30, 0.10], dtype=np.float64)

def get(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    return (np.stack([arr[i:i+T] for i in ix]),
            np.stack([arr[i+1:i+T+1] for i in ix]))

def emo_vec(e):
    return np.array([e.x, e.y, e.c[0], e.c[1]], dtype=np.float64)

def run(mode):
    np.random.seed(42)
    model = StateLM(V, d=128, n_layers=3, max_T=T, gated=True)
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(0)
    endo = LiveEndocrine(seed=0)
    emo = emo_vec(endo) if mode == "live" else FIXED
    ema = None
    t0 = time.time()
    for s in range(STEPS):
        x, y = get(tr, B, rng)
        out = model(Tensor(x), emo=emo if mode != "base" else None)
        loss = cross_entropy(out, y)
        lv = float(loss.data)
        opt.zero_grad(); loss.backward(); opt.step()
        if mode == "live":
            ema = lv if ema is None else 0.95 * ema + 0.05 * lv
            nov = float(np.clip(max(0.0, (lv - ema) / max(ema, 0.1)), 0.0, 1.0))
            endo.step(min(lv / 2.0, 3.0), nov, dt=0.2)
            endo.step(0.0, 0.0, dt=0.2)      # 空 tick
            emo = emo_vec(endo)
        if s % 100 == 0 or s == STEPS - 1:
            print(f"  [{mode:5s}] {s:3d} train {lv:.3f}  {(time.time()-t0)/(s+1)*1000:.0f}ms", flush=True)
    vrng = np.random.RandomState(12345)
    tot = 0.0
    for _ in range(10):
        x, y = get(va, B, vrng)
        out = model(Tensor(x), emo=emo if mode != "base" else None)
        tot += float(cross_entropy(out, y).data)
    return tot / 10, time.time() - t0

res = {}
for m in ("base", "fixed", "live"):
    res[m] = run(m)
print("\n=== RESULT ===")
for k, (v, t) in res.items():
    print(f"{k:6s} val {v:.4f}  {t:.0f}s")
b = res["base"][0]
for k in ("fixed", "live"):
    print(f"{k:6s} vs base {(res[k][0]-b)/b*100:+.2f}%")
