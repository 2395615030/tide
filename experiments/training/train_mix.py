"""在 269 万字符混合语料上训同一配置的模型, 看数据能否压住过拟合。"""
import sys, os, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.transformer import TinyLM
from nano_torch.data.mix import build

sys.stdout.reconfigure(encoding="utf-8")
text, ids, val_ids, keep = build()
data = np.array(ids, dtype=np.int64)
val_data = np.array(val_ids, dtype=np.int64)
V = len(keep) + 1
T, B, STEPS = 96, 32, 1200
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
print(f"train {len(data)}  val {len(val_data)}  vocab {V}", flush=True)

def get_batch(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 2, size=n)
    x = np.stack([arr[i:i+T] for i in ix])
    y = np.stack([arr[i+1:i+T+1] for i in ix])
    return x, y

def evaluate(model, rng, n=12):
    tot = 0.0
    for _ in range(n):
        x, y = get_batch(val_data, B, rng)
        tot += float(cross_entropy(model(Tensor(x), mask), y).data)
    return tot / n

np.random.seed(42)
model = TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu")
opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
rng = np.random.RandomState(0)
vrng = np.random.RandomState(999)
print(f"params {sum(p.size for p in model.params())}", flush=True)

t0 = time.time()
best = (1e9, -1)
for s in range(STEPS):
    x, y = get_batch(data, B, rng)
    loss = cross_entropy(model(Tensor(x), mask), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if s % 100 == 0 or s == STEPS - 1:
        v = evaluate(model, vrng)
        if v < best[0]:
            best = (v, s)
        dt = (time.time() - t0) / max(1, s + 1)
        print(f"step {s:4d}  train {float(loss.data):.4f}  val {v:.4f}  "
              f"{dt*1000:.0f} ms/step  best {best[0]:.4f}@{best[1]}", flush=True)

os.makedirs("D:/ROOT/Projects/DSH/Def/nano-torch/ckpt", exist_ok=True)
np.savez("D:/ROOT/Projects/DSH/Def/nano-torch/ckpt/mix_d128_l3.npz",
         **{f"p{i}": p.data for i, p in enumerate(model.params())})
print(f"done in {time.time()-t0:.0f}s, best val {best[0]:.4f} at step {best[1]}", flush=True)
