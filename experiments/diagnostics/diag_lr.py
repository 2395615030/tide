"""T=384 gated, 不同学习率, 看是不是优化不稳定。"""
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
T, B = 384, 16

def get(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    return (np.stack([arr[i:i+T] for i in ix]),
            np.stack([arr[i+1:i+T+1] for i in ix]))

def run(lr, steps=100, clip=None):
    np.random.seed(42)
    model = StateLM(V, d=128, n_layers=3, max_T=T, gated=True)
    opt = AdamW(model.params(), lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(0)
    t0 = time.time()
    losses = []
    for s in range(steps):
        x, y = get(tr, B, rng)
        loss = cross_entropy(model(Tensor(x)), y)
        opt.zero_grad(); loss.backward()
        if clip is not None:
            tot = 0.0
            for p in model.params():
                if p.grad is not None:
                    tot += float((p.grad ** 2).sum())
            nrm = np.sqrt(tot)
            if nrm > clip:
                for p in model.params():
                    if p.grad is not None:
                        p.grad *= clip / (nrm + 1e-8)
        opt.step()
        losses.append(float(loss.data))
    tail = np.mean(losses[-10:])
    print(f"lr={lr:<7} clip={clip}  first10 {np.mean(losses[:10]):.3f}  "
          f"last10 {tail:.3f}  min {min(losses):.3f}  {time.time()-t0:.0f}s", flush=True)
    return tail

run(3e-3)
run(1e-3)
run(3e-3, clip=1.0)
