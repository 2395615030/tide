"""按文档顺序采样, 状态跨块传递(截断 BPTT)。"""
import sys, os, glob, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.transformer import TinyLM
from nano_torch.data.neko import load_neko
from nano_torch.data.mix import load_tieba, TIEBA, TIEBA_VALID

T, B, K = 96, 8, 3
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)

neko = load_neko()[0]
tieba = load_tieba(TIEBA)
tieba_val = load_tieba(TIEBA_VALID)
chars = sorted(set(neko + tieba))
vocab = {ch: i + 3 for i, ch in enumerate(chars)}
V = len(vocab) + 3

def enc(s):
    return np.array([vocab.get(c, 0) for c in s], dtype=np.int64)

# 文档: 19 个 neko 文件各算一个, tieba 切成 20 万字符的段
docs, doc_kind = [], []
for f in sorted(glob.glob(r"D:\ROOT\Datasets\Neko-A\*.json")):
    import json
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    s = ""
    for t in d.get("conversation", []):
        s += (t.get("content") or "") + "\n"
    if len(s) > T * K + 10:
        docs.append(enc(s)); doc_kind.append(0)
tb = enc(tieba)
step = 200000
for i in range(0, len(tb) - step, step):
    docs.append(tb[i:i+step]); doc_kind.append(1)
print(f"docs {len(docs)}  (neko {doc_kind.count(0)}, tieba {doc_kind.count(1)}), vocab {V}", flush=True)

def one_step(model, opt, rng, use_state):
    di = rng.randint(0, len(docs), size=B)
    st = [rng.randint(0, len(docs[i]) - T*K - 2) for i in di]
    h = None
    tot = 0.0
    for k in range(K):
        x = np.stack([docs[i][s + k*T: s + (k+1)*T] for i, s in zip(di, st)])
        y = np.stack([docs[i][s + k*T + 1: s + (k+1)*T + 1] for i, s in zip(di, st)])
        if use_state:
            logits, h = model(Tensor(x), mask, h=h, return_state=True)
        else:
            logits = model(Tensor(x), mask)
        loss = cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if use_state:
            h = Tensor(h.data.copy())   # 截断 BPTT
        tot += float(loss.data)
    return tot / K

np.random.seed(42)
model = TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu", use_state=True)
opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
rng = np.random.RandomState(0)
print(f"params {sum(p.size for p in model.params())}", flush=True)

t0 = time.time()
for s in range(20):
    l = one_step(model, opt, rng, True)
    if s % 5 == 0:
        print(f"  step {s:3d}  loss {l:.4f}  {(time.time()-t0)/(s+1)*1000:.0f} ms/step", flush=True)
print("smoke ok", flush=True)
