import sys, os, time, cProfile, pstats, io
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.transformer import TinyLM

np.random.seed(0)
V, T, B, D = 512, 64, 32, 128
model = TinyLM(V, d=D, n_heads=4, n_layers=2, ff=512, max_T=T)
opt = AdamW(model.params(), lr=3e-3)
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
x = np.random.randint(0, V, size=(B, T))
y = np.random.randint(0, V, size=(B, T))

def one_step():
    logits = model(Tensor(x), mask)
    loss = cross_entropy(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()

for _ in range(2):
    one_step()
t0 = time.time()
for _ in range(5):
    one_step()
print(f"per step: {(time.time()-t0)/5*1000:.0f} ms  (B={B}, T={T}, d={D})")

pr = cProfile.Profile()
pr.enable()
for _ in range(3):
    one_step()
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
print(s.getvalue()[:4000])
