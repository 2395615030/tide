"""公平 A/B: 同一进程、同一语料、同一 seed。"""
import sys, os, glob, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.transformer import TinyLM

ROOT = "D:/ROOT/Projects/DSH/Def/nano-torch"
files = sorted(glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True))
text = "\n".join(open(f, encoding="utf-8").read() for f in files)
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int64)
split = int(len(data) * 0.9)
train_data, val_data = data[:split], data[split:]
T, B, STEPS = 64, 32, 300
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
print(f"corpus {len(text)} chars, vocab {len(chars)}, files {len(files)}")

BASE_LR = 3e-3

def run(mode):
    np.random.seed(42)
    model = TinyLM(len(chars), d=128, n_heads=4, n_layers=2, ff=512, max_T=T)
    opt = AdamW(model.params(), lr=BASE_LR, weight_decay=0.01)
    prev, ema_d, e = None, 0.0, 0.0
    lrs = []
    for s in range(STEPS):
        ix = np.random.randint(0, len(train_data) - T - 1, size=B)
        x = np.stack([train_data[i:i+T] for i in ix])
        y = np.stack([train_data[i+1:i+T+1] for i in ix])
        loss = cross_entropy(model(Tensor(x), mask), y)
        lv = float(loss.data)
        if mode == "emotion" and prev is not None:
            d = lv - prev
            surprise = d - ema_d
            ema_d = 0.9 * ema_d + 0.1 * d
            e = 0.9 * e + 0.1 * np.tanh(5.0 * surprise)
            lr = BASE_LR * float(np.clip(1.0 + 0.3 * e, 0.5, 1.5))
        else:
            lr = BASE_LR
        prev = lv
        lrs.append(lr)
        opt.zero_grad(); loss.backward(); opt.step(lr=lr)
    # 用同一批验证样本
    rng = np.random.RandomState(999)
    tot = 0.0
    for _ in range(16):
        ix = rng.randint(0, len(val_data) - T - 1, size=B)
        x = np.stack([val_data[i:i+T] for i in ix])
        y = np.stack([val_data[i+1:i+T+1] for i in ix])
        tot += float(cross_entropy(model(Tensor(x), mask), y).data)
    return tot / 16, np.mean(lrs), np.std(lrs)

t0 = time.time()
v_fixed, m1, s1 = run("fixed")
v_emo, m2, s2 = run("emotion")
print(f"\nfixed   : val {v_fixed:.4f}  lr {m1:.6f} +- {s1:.6f}")
print(f"emotion : val {v_emo:.4f}  lr {m2:.6f} +- {s2:.6f}")
print(f"delta   : {(v_emo - v_fixed) / v_fixed * 100:+.2f}%   ({time.time()-t0:.0f}s)")
