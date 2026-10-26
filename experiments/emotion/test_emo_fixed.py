"""对照: emo 固定不变。检验 -4.72% 是"偏置"还是"情绪的动态"。"""
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
split = int(len(data) * 0.9)
tr, va = data[:split], data[split:]
V = len(keep) + 1
T, B, STEPS = 96, 32, 300

def get(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    return (np.stack([arr[i:i+T] for i in ix]),
            np.stack([arr[i+1:i+T+1] for i in ix]))

def run(mode):
    np.random.seed(42)
    model = StateLM(V, d=128, n_layers=3, max_T=T, gated=True)
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(0)
    emo = np.array([0.30, 0.10, 0.00, 0.50])     # 固定, 永远不变
    t0 = time.time()
    for s in range(STEPS):
        x, y = get(tr, B, rng)
        out = model(Tensor(x), emo=emo if mode == "fixed" else None)
        loss = cross_entropy(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
    # 验证
    vrng = np.random.RandomState(12345)
    tot = 0.0
    for _ in range(10):
        x, y = get(va, B, vrng)
        out = model(Tensor(x), emo=emo if mode == "fixed" else None)
        tot += float(cross_entropy(out, y).data)
    return tot / 10, time.time() - t0

v, t = run("fixed")
print(f"fixed-emo val {v:.4f}  {t:.0f}s")
print(f"compare: base 3.3716 | dynamic-emo 3.2124")
print(f"fixed vs base: {(v-3.3716)/3.3716*100:+.2f}%")
print(f"fixed vs dyn : {(v-3.2124)/3.2124*100:+.2f}%")
