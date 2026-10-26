"""字符级 LM + 情绪调制对比。语料用项目自己的源码(自举)。"""
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
print(f"corpus: {len(text)} chars, vocab {len(chars)}, files {len(files)}")

split = int(len(data) * 0.9)
train_data, val_data = data[:split], data[split:]
T, B = 64, 32
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)

def batch(arr, n):
    ix = np.random.randint(0, len(arr) - T - 1, size=n)
    x = np.stack([arr[i:i+T] for i in ix])
    y = np.stack([arr[i+1:i+T+1] for i in ix])
    return x, y

def evaluate(model, arr, n=8):
    tot = 0.0
    for _ in range(n):
        x, y = batch(arr, B)
        logits = model(Tensor(x), mask)
        loss = cross_entropy(logits, y)
        tot += float(loss.data)
    return tot / n

def run(use_emotion, steps=300, lr=3e-3):
    np.random.seed(42)
    model = TinyLM(len(chars), d=128, n_heads=4, n_layers=2, ff=512,
                   max_T=T, use_emotion=use_emotion)
    opt = AdamW(model.params(), lr=lr, weight_decay=0.01)
    nparam = sum(p.size for p in model.params())
    t0 = time.time()
    for s in range(steps):
        x, y = batch(train_data, B)
        logits = model(Tensor(x), mask)
        loss = cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0 or s == steps - 1:
            print(f"  [{'EMO' if use_emotion else 'BASE'}] step {s:4d}  "
                  f"train {float(loss.data):.4f}  val {evaluate(model, val_data):.4f}")
    vt = evaluate(model, val_data, 16)
    return vt, nparam, time.time() - t0

print("\n--- baseline ---")
v_base, p_base, t_base = 2.2160, 0, 0.0  # from previous run
print("\n--- with emotion modulation ---")
v_emo, p_emo, t_emo = run(True)

print("\n=== RESULT ===")
print(f"baseline : val {v_base:.4f}  params {p_base}  {t_base:.1f}s")
print(f"emotion  : val {v_emo:.4f}  params {p_emo}  {t_emo:.1f}s  (+{p_emo-p_base} params)")
print(f"delta    : {(v_emo - v_base)/v_base*100:+.2f}%  val loss")
