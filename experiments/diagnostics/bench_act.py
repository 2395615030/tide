import sys, os, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.transformer import TinyLM

np.random.seed(0)
V, T, B = 512, 64, 32
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
x = np.random.randint(0, V, size=(B, T))
y = np.random.randint(0, V, size=(B, T))

def bench(act, d, layers, steps=40):
    np.random.seed(0)
    model = TinyLM(V, d=d, n_heads=4, n_layers=layers, ff=4*d, max_T=T, act=act)
    opt = AdamW(model.params(), lr=1e-3)
    for _ in range(2):
        loss = cross_entropy(model(Tensor(x), mask), y)
        opt.zero_grad(); loss.backward(); opt.step()
    t0 = time.time()
    lst = []
    for _ in range(steps):
        loss = cross_entropy(model(Tensor(x), mask), y)
        opt.zero_grad(); loss.backward(); opt.step()
        lst.append(float(loss.data))
    dt = (time.time() - t0) / steps
    npar = sum(p.size for p in model.params())
    print(f"{act:5s} d={d:3d} L={layers}  {dt*1000:6.0f} ms/step  params {npar:>8}  loss {lst[-1]:.3f}")

bench("gelu", 128, 2)
bench("relu", 128, 2)
bench("silu", 128, 2)
bench("relu", 256, 3)
