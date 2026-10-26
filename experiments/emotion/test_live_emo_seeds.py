"""重复验证: 换 seed 再跑 fixed vs live, 看 1.6% 是否稳定。"""
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

def get(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    return (np.stack([arr[i:i+T] for i in ix]),
            np.stack([arr[i+1:i+T+1] for i in ix]))

def emo_vec(e):
    return np.array([e.x, e.y, e.c[0], e.c[1]], dtype=np.float64)

def run(mode, seed):
    np.random.seed(seed)
    model = StateLM(V, d=128, n_layers=3, max_T=T, gated=True)
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(seed * 7 + 1)
    endo = LiveEndocrine(seed=seed)
    fixed = np.array([0.6, 0.0, 0.30, 0.10], dtype=np.float64)
    emo = emo_vec(endo) if mode == "live" else fixed
    ema = None
    for s in range(STEPS):
        x, y = get(tr, B, rng)
        out = model(Tensor(x), emo=emo)
        loss = cross_entropy(out, y)
        lv = float(loss.data)
        opt.zero_grad(); loss.backward(); opt.step()
        if mode == "live":
            ema = lv if ema is None else 0.95 * ema + 0.05 * lv
            nov = float(np.clip(max(0.0, (lv - ema) / max(ema, 0.1)), 0.0, 1.0))
            endo.step(min(lv / 2.0, 3.0), nov, dt=0.2)
            endo.step(0.0, 0.0, dt=0.2)
            emo = emo_vec(endo)
    vrng = np.random.RandomState(12345)
    tot = 0.0
    for _ in range(10):
        x, y = get(va, B, vrng)
        tot += float(cross_entropy(model(Tensor(x), emo=emo), y).data)
    return tot / 10

for seed in (1, 2):
    f = run("fixed", seed)
    l = run("live", seed)
    print(f"seed {seed}: fixed {f:.4f}  live {l:.4f}  delta {(l-f)/f*100:+.2f}%", flush=True)
