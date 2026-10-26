"""非平稳任务: 正序 200 步 -> 突然切到反序 200 步。
常数模型会灾难性遗忘; 情绪模型若能被"惊讶"触发, 应在切换后恢复更快。
"""
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
tr = data[:int(len(data)*0.9)]
V = len(keep) + 1
T, B = 96, 32
SWITCH = 200
STEPS = 400

def get(rng):
    ix = rng.randint(0, len(tr) - T - 2, size=B)
    x = np.stack([tr[i:i+T] for i in ix])
    fwd = np.stack([tr[i+1:i+T+1] for i in ix])
    bwd = np.concatenate([x[:, :1], x[:, :-1]], axis=1)
    return x, fwd, bwd

def run(use_emo):
    np.random.seed(42)
    model = StateLM(V, d=128, n_layers=3, max_T=T, gated=True)
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(0)
    endo = Endocrine(); emo = endo.c0.copy(); ema = None
    losses = []
    t0 = time.time()
    for s in range(STEPS):
        x, fwd, bwd = get(rng)
        y = fwd if s < SWITCH else bwd
        out = model(Tensor(x), emo=emo if use_emo else None)
        loss = cross_entropy(out, y)
        lv = float(loss.data)
        opt.zero_grad(); loss.backward(); opt.step()
        if use_emo:
            ema = lv if ema is None else 0.95 * ema + 0.05 * lv
            novelty = float(np.clip(max(0.0, (lv - ema) / max(ema, 0.1)), 0.0, 1.0))
            emo = endo.step(min(lv / 2.0, 3.0), novelty)
        losses.append(lv)
        if s % 25 == 0 or s in (SWITCH - 1, SWITCH):
            stage = "A(fwd)" if s < SWITCH else "B(bwd)"
            print(f"  [{'emo ' if use_emo else 'base'}] {s:3d} {stage} loss {lv:.3f} "
                  f"emo[{emo[0]:.2f} {emo[1]:.2f} {emo[2]:.2f} {emo[3]:.2f}]", flush=True)
    L = np.array(losses)
    early = L[SWITCH:SWITCH+30].mean()
    late = L[SWITCH+100:].mean()
    print(f"  --> after-switch first30 {early:.3f}   settled {late:.3f}", flush=True)
    return early, late, L[SWITCH-20:SWITCH].mean(), time.time() - t0

print("=== baseline ===")
b_early, b_late, b_pre, b_t = run(False)
print("\n=== with endocrine state ===")
e_early, e_late, e_pre, e_t = run(True)

print("\n=== RESULT ===")
print(f"base : pre-switch {b_pre:.3f}  first30 {b_early:.3f}  settled {b_late:.3f}  {b_t:.0f}s")
print(f"emo  : pre-switch {e_pre:.3f}  first30 {e_early:.3f}  settled {e_late:.3f}  {e_t:.0f}s")
print(f"adaptation speed: {(e_early-b_early)/b_early*100:+.1f}%  (negative = 恢复更快)")
print(f"final quality   : {(e_late-b_late)/b_late*100:+.1f}%")
