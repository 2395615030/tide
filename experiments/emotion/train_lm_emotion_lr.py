"""情绪 -> 学习率: 带记忆的惊讶状态, 只调制更新幅度, 不碰 logits。"""
import sys, os, glob, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.transformer import TinyLM

ROOT = "D:/ROOT/Projects/DSH/Def/nano-torch"
files = glob.glob(os.path.join(ROOT, "**", "*.py"), recursive=True)
text = "\n".join(open(f, encoding="utf-8").read() for f in files)
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int64)
split = int(len(data) * 0.9)
train_data, val_data = data[:split], data[split:]
T, B = 64, 32
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)

def batch(arr, n):
    ix = np.random.randint(0, len(arr) - T - 1, size=n)
    return (np.stack([arr[i:i+T] for i in ix]),
            np.stack([arr[i+1:i+T+1] for i in ix]))

def evaluate(model, arr, n=8):
    tot = 0.0
    for _ in range(n):
        x, y = batch(arr, B)
        tot += float(cross_entropy(model(Tensor(x), mask), y).data)
    return tot / n

def run(mode, steps=300, base_lr=3e-3):
    """mode: 'fixed' | 'emotion'"""
    np.random.seed(42)
    model = TinyLM(len(chars), d=128, n_heads=4, n_layers=2, ff=512, max_T=T)
    opt = AdamW(model.params(), lr=base_lr, weight_decay=0.01)

    e = 0.0          # 情绪状态(带记忆)
    avg = None       # 惊讶基线
    lrs = []
    t0 = time.time()
    for s in range(steps):
        x, y = batch(train_data, B)
        loss = cross_entropy(model(Tensor(x), mask), y)
        lv = float(loss.data)

        if mode == "emotion":
            if avg is None:
                avg = lv
            surprise = lv - avg
            avg = 0.99 * avg + 0.01 * lv
            e = 0.9 * e + 0.1 * np.tanh(2.0 * surprise)
            lr = base_lr * float(np.clip(1.0 + 0.5 * e, 0.3, 2.0))
        else:
            lr = base_lr
        lrs.append(lr)

        opt.zero_grad(); loss.backward(); opt.step(lr=lr)
        if s % 100 == 0 or s == steps - 1:
            print(f"  [{mode:7s}] step {s:4d}  train {lv:.4f}  "
                  f"val {evaluate(model, val_data):.4f}  lr {lr:.5f}  e {e:+.3f}")
    vt = evaluate(model, val_data, 16)
    return vt, np.mean(lrs), np.std(lrs), time.time() - t0

print("--- fixed lr ---")
v1, m1, s1, t1 = run("fixed")
print("\n--- emotion-driven lr ---")
v2, m2, s2, t2 = run("emotion")

print("\n=== RESULT ===")
print(f"fixed   : val {v1:.4f}  lr mean {m1:.5f} std {s1:.5f}  {t1:.0f}s")
print(f"emotion : val {v2:.4f}  lr mean {m2:.5f} std {s2:.5f}  {t2:.0f}s")
print(f"delta   : {(v2 - v1) / v1 * 100:+.2f}%")
