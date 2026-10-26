"""3 seed 验证: attention vs 门控状态 (T=96, 300 步)。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.statelm import StateLM
from nano_torch.nn.transformer import TinyLM
from nano_torch.data.neko import load_neko

text, ids, keep = load_neko()
data = np.array(ids, dtype=np.int64)
split = int(len(data) * 0.9)
tr, va = data[:split], data[split:]
V = len(keep) + 1
T, B, STEPS = 96, 32, 300
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)

def get(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    return (np.stack([arr[i:i+T] for i in ix]),
            np.stack([arr[i+1:i+T+1] for i in ix]))

def run(kind, seed):
    np.random.seed(seed)
    if kind == "attn":
        model = TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu")
    else:
        model = StateLM(V, d=128, n_layers=3, max_T=T, gated=True)
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(seed * 13 + 5)
    for s in range(STEPS):
        x, y = get(tr, B, rng)
        out = model(Tensor(x), mask) if kind == "attn" else model(Tensor(x))
        loss = cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    vrng = np.random.RandomState(12345)
    tot = 0.0
    for _ in range(10):
        x, y = get(va, B, vrng)
        out = model(Tensor(x), mask) if kind == "attn" else model(Tensor(x))
        tot += float(cross_entropy(out, y).data)
    return tot / 10

t0 = time.time()
for seed in (0, 1, 2):
    a = run("attn", seed)
    g = run("gated", seed)
    print(f"seed {seed}: attn {a:.4f}  gated {g:.4f}  delta {(g-a)/a*100:+.2f}%  "
          f"[{time.time()-t0:.0f}s]", flush=True)
