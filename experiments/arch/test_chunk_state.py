"""跨块连续训练: 有状态 vs 无状态。验收看每块位置的 loss。"""
import sys, os, glob, json, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.transformer import TinyLM
from nano_torch.data.mix import load_tieba, TIEBA

T, B, K, STEPS = 96, 8, 3, 600
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)

neko = "".join(
    "".join((t.get("content") or "") + "\n" for t in json.load(open(f, encoding="utf-8")).get("conversation", []))
    for f in sorted(glob.glob(r"D:\ROOT\Datasets\Neko-A\*.json"))
)
tieba = load_tieba(TIEBA)
chars = sorted(set(neko + tieba))
vocab = {ch: i + 3 for i, ch in enumerate(chars)}
V = len(vocab) + 3

enc = lambda s: np.array([vocab.get(c, 0) for c in s], dtype=np.int64)

docs, kind = [], []
for f in sorted(glob.glob(r"D:\ROOT\Datasets\Neko-A\*.json")):
    s = "".join((t.get("content") or "") + "\n" for t in json.load(open(f, encoding="utf-8")).get("conversation", []))
    if len(s) > T*K + 10:
        docs.append(enc(s)); kind.append(0)
tb = enc(tieba)
for i in range(0, len(tb) - 200000, 200000):
    docs.append(tb[i:i+200000]); kind.append(1)
kinds = np.array(kind)
print(f"docs {len(docs)} (neko {(kinds==0).sum()}, tieba {(kinds==1).sum()}), vocab {V}", flush=True)

def train_step(model, opt, rng, use_state):
    di = rng.randint(0, len(docs), size=B)
    st = [rng.randint(0, len(docs[i]) - T*K - 2) for i in di]
    h = None
    tot = 0.0
    for k in range(K):
        x = np.stack([docs[i][s+k*T: s+(k+1)*T] for i, s in zip(di, st)])
        y = np.stack([docs[i][s+k*T+1: s+(k+1)*T+1] for i, s in zip(di, st)])
        if use_state:
            logits, h = model(Tensor(x), mask, h=h, return_state=True)
        else:
            logits = model(Tensor(x), mask)
        loss = cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if use_state:
            h = Tensor(h.data.copy())
        tot += float(loss.data)
    return tot / K

def evaluate(model, use_state, rng, n=4):
    out = {"neko": dict(), "tieba": dict()}
    for dom, name in [(0, "neko"), (1, "tieba")]:
        ids = np.where(kinds == dom)[0]
        per = {k: 0.0 for k in range(K)}
        for _ in range(n):
            di = rng.choice(ids, size=B)
            st = [rng.randint(0, len(docs[i]) - T*K - 2) for i in di]
            h = None
            for k in range(K):
                x = np.stack([docs[i][s+k*T: s+(k+1)*T] for i, s in zip(di, st)])
                y = np.stack([docs[i][s+k*T+1: s+(k+1)*T+1] for i, s in zip(di, st)])
                if use_state:
                    logits, h = model(Tensor(x), mask, h=h, return_state=True)
                else:
                    logits = model(Tensor(x), mask)
                per[k] += float(cross_entropy(logits, y).data)
                if use_state:
                    h = Tensor(h.data.copy())
        out[name] = {k: v / n for k, v in per.items()}
    return out

def run(use_state):
    np.random.seed(42)
    model = TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu", use_state=use_state)
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(0); vrng = np.random.RandomState(999)
    t0 = time.time()
    for s in range(STEPS):
        l = train_step(model, opt, rng, use_state)
        if s % 200 == 0 or s == STEPS - 1:
            e = evaluate(model, use_state, vrng)
            print(f"  [{'state' if use_state else 'plain'}] {s:3d} train {l:.3f}  "
                  f"neko@{['b0','b1','b2']} " + " ".join(f"{e['neko'][k]:.3f}" for k in range(K)) +
                  f"  tieba " + " ".join(f"{e['tieba'][k]:.3f}" for k in range(K)) +
                  f"  {(time.time()-t0)/(s+1)*1000:.0f}ms", flush=True)
    return evaluate(model, use_state, vrng, n=8), time.time() - t0

print("\n--- plain (no state) ---")
e1, d1 = run(False)
print("\n--- with state ---")
e2, d2 = run(True)

print("\n=== RESULT (loss per chunk position) ===")
for name in ("neko", "tieba"):
    print(f"{name}: plain " + " ".join(f"b{k}={e1[name][k]:.4f}" for k in range(K)))
    print(f"{name}: state " + " ".join(f"b{k}={e2[name][k]:.4f}" for k in range(K)))
    print(f"{name}: delta " + " ".join(f"b{k}={(e2[name][k]-e1[name][k])/e1[name][k]*100:+.1f}%" for k in range(K)))
print(f"times: plain {d1:.0f}s  state {d2:.0f}s")
