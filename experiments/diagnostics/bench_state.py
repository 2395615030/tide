import sys, os, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.statelm import StateLM
from nano_torch.nn.transformer import TinyLM

T, B, V = 96, 16, 512
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
x = np.random.randint(0, V, size=(B, T))
y = np.random.randint(0, V, size=(B, T))

def bench(model, name, use_mask, steps=15):
    opt = AdamW(model.params(), lr=1e-3)
    for _ in range(2):
        out = model(Tensor(x), mask) if use_mask else model(Tensor(x))
        loss = cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    t0 = time.time()
    for _ in range(steps):
        out = model(Tensor(x), mask) if use_mask else model(Tensor(x))
        loss = cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    dt = (time.time() - t0) / steps
    print(f"{name:14s} {dt*1000:6.0f} ms/step  params {sum(p.size for p in model.params()):>9}  "
          f"loss {float(loss.data):.3f}", flush=True)

np.random.seed(0)
bench(TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, act="silu"), "transformer", True)
np.random.seed(0)
bench(StateLM(V, d=128, n_layers=3), "state (no attn)", False)
