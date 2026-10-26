import sys, os, time, cProfile, pstats, io
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.statelm import StateLM
from nano_torch.data.neko import load_neko

text, ids, keep = load_neko()
V = len(keep) + 1
T, B = 384, 16
np.random.seed(0)
model = StateLM(V, d=128, n_layers=3, max_T=T, gated=True)
opt = AdamW(model.params(), lr=1e-3)
x = np.random.randint(0, V, size=(B, T))
y = np.random.randint(0, V, size=(B, T))

def step():
    out = model(Tensor(x))
    loss = cross_entropy(out, y)
    opt.zero_grad(); loss.backward(); opt.step()

step()
t0 = time.time()
for _ in range(3): step()
print(f"per step: {(time.time()-t0)/3*1000:.0f} ms")

pr = cProfile.Profile(); pr.enable()
for _ in range(3): step()
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(15)
print(s.getvalue()[:3500])
