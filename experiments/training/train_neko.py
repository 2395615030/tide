"""在 Neko-A 对话数据上训一个小对话模型。"""
import sys, os, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.transformer import TinyLM
from nano_torch.data.neko import load_neko

text, ids, keep = load_neko()
data = np.array(ids, dtype=np.int64)
split = int(len(data) * 0.9)
train_data, val_data = data[:split], data[split:]
V = len(keep) + 1
T, B = 96, 32
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
print(f"tokens {len(data)}, vocab {V}, train {len(train_data)}, val {len(val_data)}")

def get_batch(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    x = np.stack([arr[i:i+T] for i in ix])
    y = np.stack([arr[i+1:i+T+1] for i in ix])
    return x, y

def evaluate(model, rng, n=10):
    tot = 0.0
    for _ in range(n):
        x, y = get_batch(val_data, B, rng)
        tot += float(cross_entropy(model(Tensor(x), mask), y).data)
    return tot / n

np.random.seed(42)
model = TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu")
opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
rng = np.random.RandomState(0)
nparam = sum(p.size for p in model.params())
print(f"params {nparam}")

t0 = time.time()
BEST = None
for s in range(800):
    x, y = get_batch(train_data, B, rng)
    loss = cross_entropy(model(Tensor(x), mask), y)
    opt.zero_grad(); loss.backward(); opt.step()
    if s % 100 == 0 or s == 799:
        v = evaluate(model, rng)
        print(f"step {s:4d}  train {float(loss.data):.4f}  val {v:.4f}  "
              f"{(time.time()-t0)/max(1,s)*1000:.0f} ms/step", flush=True)

# 保存参数
os.makedirs("D:/ROOT/Projects/DSH/Def/nano-torch/ckpt", exist_ok=True)
np.savez("D:/ROOT/Projects/DSH/Def/nano-torch/ckpt/neko_d128_l3.npz",
         **{f"p{i}": p.data for i, p in enumerate(model.params())})
print(f"done in {time.time()-t0:.0f}s, saved ckpt")
