"""情绪调制门控: 无情绪 vs 内分泌驱动。"""
import sys, os, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.core.endocrine import Endocrine
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

def evaluate(model, emo_mode, n=6):
    rng = np.random.RandomState(12345)
    endo = Endocrine(); emo = endo.c0.copy()
    tot = 0.0
    for _ in range(n):
        x, y = get(va, B, rng)
        out = model(Tensor(x), emo=emo if emo_mode else None)
        tot += float(cross_entropy(out, y).data)
        if emo_mode:
            emo = endo.step(5.0, 0.0)   # 验证时保持稳定刺激
    return tot / n

def run(use_emo):
    np.random.seed(42)
    model = StateLM(V, d=128, n_layers=3, max_T=T, gated=True)
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(0)
    endo = Endocrine(); emo = endo.c0.copy()
    ema = None
    t0 = time.time()
    emo_trace = []
    for s in range(STEPS):
        x, y = get(tr, B, rng)
        out = model(Tensor(x), emo=emo if use_emo else None)
        loss = cross_entropy(out, y)
        lv = float(loss.data)
        opt.zero_grad(); loss.backward(); opt.step()
        if use_emo:
            ema = lv if ema is None else 0.98 * ema + 0.02 * lv
            novelty = float(np.clip(max(0.0, (lv - ema) / max(ema, 0.1)), 0.0, 1.0))
            emo = endo.step(min(lv / 2.0, 3.0), novelty)
            emo_trace.append(emo.copy())
        if s % 100 == 0 or s == STEPS - 1:
            v = evaluate(model, use_emo)
            e = " ".join(f"{c:.2f}" for c in emo) if use_emo else "-"
            print(f"  [{'emo' if use_emo else 'base'}] {s:3d} train {lv:.3f}  val {v:.3f}  "
                  f"emo[{e}]  {(time.time()-t0)/(s+1)*1000:.0f}ms", flush=True)
    if emo_trace:
        tr_arr = np.array(emo_trace)
        print(f"  emo range: arousal {tr_arr[:,0].min():.2f}-{tr_arr[:,0].max():.2f}  "
              f"stress {tr_arr[:,1].min():.2f}-{tr_arr[:,1].max():.2f}  "
              f"novelty {tr_arr[:,2].min():.2f}-{tr_arr[:,2].max():.2f}  "
              f"satiety {tr_arr[:,3].min():.2f}-{tr_arr[:,3].max():.2f}", flush=True)
    return evaluate(model, use_emo, n=10), time.time() - t0

b, tb = run(False)
e, te = run(True)
print(f"\nbase val {b:.4f}  {tb:.0f}s")
print(f"emo  val {e:.4f}  {te:.0f}s   delta {(e-b)/b*100:+.2f}%")
