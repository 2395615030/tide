"""诊断 seed1: gated 在 500 步时是否发散? 加 clip 能否修复。"""
import sys, os, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.statelm import StateLM
from nano_torch.data.neko import load_neko

text, ids, keep = load_neko()
data = np.array(ids, dtype=np.int64)
tr = data[:int(len(data)*0.9)]
V = len(keep) + 1
T, B, STEPS = 96, 32, 500

def get(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    return (np.stack([arr[i:i+T] for i in ix]),
            np.stack([arr[i+1:i+T+1] for i in ix]))

def clip(ps, c):
    tot = 0.0
    for p in ps:
        if p.grad is not None:
            tot += float((p.grad ** 2).sum())
    nrm = np.sqrt(tot)
    if nrm > c:
        for p in ps:
            if p.grad is not None:
                p.grad *= c / (nrm + 1e-8)
    return nrm

def run(clip_val, seed=1):
    np.random.seed(seed)
    model = StateLM(V, d=128, n_layers=3, max_T=T, gated=True)
    ps = model.params()
    opt = AdamW(ps, lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(seed * 13 + 5)
    t0 = time.time()
    for s in range(STEPS):
        x, y = get(tr, B, rng)
        loss = cross_entropy(model(Tensor(x)), y)
        opt.zero_grad(); loss.backward()
        nrm = clip(ps, clip_val) if clip_val else -1
        opt.step()
        if s % 100 == 0 or s == STEPS - 1:
            print(f"  clip={clip_val} step {s:3d} loss {float(loss.data):.3f} "
                  f"gradnorm {nrm:.2f} {(time.time()-t0)/(s+1)*1000:.0f}ms", flush=True)
    return float(loss.data)

print("=== no clip ===")
run(None)
print("\n=== clip=1.0 ===")
run(1.0)
