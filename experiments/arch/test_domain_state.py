"""域标记实验: 可见标记 vs 纯状态注入。

基线: 输入首位是真实域标记 \x01/\x02
实验: 输入首位是占位 \x00, 域信息只通过 state 向量进入每层
验收: 两个域各自算 val loss
"""
import sys, os, glob, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from nano_torch.core.tensor import Tensor, cross_entropy
from nano_torch.core.optim import AdamW
from nano_torch.nn.transformer import TinyLM
from nano_torch.data.neko import load_neko
from nano_torch.data.mix import load_tieba, TIEBA, TIEBA_VALID

PAD, MN, MT = "\x00", "\x01", "\x02"
T, B, STEPS = 96, 32, 600
mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)

neko = load_neko()[0]
tieba = load_tieba(TIEBA)
tieba_val = load_tieba(TIEBA_VALID)

chars = sorted(set(neko + tieba))
vocab = {ch: i + 3 for i, ch in enumerate(chars)}
vocab[PAD], vocab[MN], vocab[MT] = 0, 1, 2
V = len(vocab) + 3
print(f"vocab {V}, neko {len(neko)}, tieba {len(tieba)}, tieba_val {len(tieba_val)}")

def enc(s):
    return np.array([vocab.get(c, 0) for c in s], dtype=np.int64)

neko_ids = enc(neko)
tieba_ids = enc(tieba)
tv_ids = enc(tieba_val)
neko_val = neko_ids[int(len(neko_ids)*0.9):]
tieba_val_ids = tv_ids

def windows(arr, n, rng):
    ix = rng.randint(0, len(arr) - T - 1, size=n)
    return np.stack([arr[i:i+T] for i in ix]), np.stack([arr[i+1:i+T+1] for i in ix])

def make_batch(mode, rng, n=B):
    half = n // 2
    xa, ya = windows(neko_ids, half, rng)
    xb, yb = windows(tieba_ids, half, rng)
    if mode == "mark":
        xa = np.concatenate([[np.full((half,1), vocab[MN])]], 0) if False else np.hstack([np.full((half,1), vocab[MN]), xa[:, :-1]])
        xb = np.hstack([np.full((half,1), vocab[MT]), xb[:, :-1]])
    else:
        xa = np.hstack([np.zeros((half,1), dtype=np.int64), xa[:, :-1]])
        xb = np.hstack([np.zeros((half,1), dtype=np.int64), xb[:, :-1]])
    x = np.vstack([xa, xb])
    y = np.vstack([ya, yb])
    dom = np.array([0]*half + [1]*half)
    return x, y, dom

def evaluate(model, mode, rng, n=8):
    out = []
    for arr in (neko_val, tieba_val_ids):
        tot = 0.0
        for _ in range(n):
            ix = rng.randint(0, len(arr) - T - 1, size=B)
            x = np.stack([arr[i:i+T] for i in ix])
            y = np.stack([arr[i+1:i+T+1] for i in ix])
            dom = np.zeros(B, dtype=np.int64) if arr is neko_val else np.ones(B, dtype=np.int64)
            if mode == "mark":
                first = vocab[MN] if arr is neko_val else vocab[MT]
                x = np.hstack([np.full((B,1), first), x[:, :-1]])
            else:
                x = np.hstack([np.zeros((B,1), dtype=np.int64), x[:, :-1]])
            d = dom if mode == "state" else None
            tot += float(cross_entropy(model(Tensor(x), mask, d), y).data)
        out.append(tot / n)
    return out  # [neko_val, tieba_val]

def run(mode):
    np.random.seed(42)
    use_state = (mode == "state")
    model = TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu", use_state=use_state)
    opt = AdamW(model.params(), lr=3e-3, weight_decay=0.01)
    rng = np.random.RandomState(0)
    vrng = np.random.RandomState(999)
    t0 = time.time()
    for s in range(STEPS):
        x, y, dom = make_batch(mode, rng)
        d = dom if use_state else None
        loss = cross_entropy(model(Tensor(x), mask, d), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 200 == 0 or s == STEPS - 1:
            nv, tv = evaluate(model, mode, vrng)
            print(f"  [{mode:5s}] step {s:3d} train {float(loss.data):.4f}  "
                  f"neko_val {nv:.4f}  tieba_val {tv:.4f}  {(time.time()-t0)/max(1,s)*1000:.0f} ms/step", flush=True)
    return evaluate(model, mode, vrng, n=12), time.time() - t0

print("\n--- baseline: visible domain marker ---")
(n1, t1), d1 = run("mark")
print("\n--- experiment: state injection only ---")
(n2, t2), d2 = run("state")

print("\n=== RESULT ===")
print(f"mark  : neko {n1:.4f}  tieba {t1:.4f}  {d1:.0f}s")
print(f"state : neko {n2:.4f}  tieba {t2:.4f}  {d2:.0f}s")
print(f"delta : neko {(n2-n1)/n1*100:+.1f}%  tieba {(t2-t1)/t1*100:+.1f}%")
