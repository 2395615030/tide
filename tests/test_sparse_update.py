"""对比: 每步全量更新 vs 每步只更新 1/4 的块。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from nano_torch.core.tensor import Tensor, gelu, cross_entropy
from nano_torch.core.store import save_paged, PagedMatrix, CachePool
from nano_torch.nn.paged import PagedLinear

D = "D:/ROOT/Projects/DSH/Def/nano-torch/data"
IN, H, OUT, CHUNK, POOL = 256, 1024, 16, 64, 6 * 1024 * 1024
specs = [("W1", (H, IN), 0.05), ("W2", (H, H), 0.02), ("W3", (OUT, H), 0.02)]
N, B, EPOCHS = 512, 32, 18

def build():
    np.random.seed(0)
    for name, shape, s in specs:
        W = (np.random.randn(*shape) * s).astype(np.float32)
        save_paged(os.path.join(D, name + ".ndst"), W, CHUNK)

def train(groups):
    build()
    np.random.seed(1)
    X = np.random.randn(N, IN).astype(np.float32)
    Y = X[:, :OUT].argmax(1).astype(np.int64)
    pool = CachePool(POOL)
    pms = [PagedMatrix(os.path.join(D, n + ".ndst"), pool, writable=True) for n, _, _ in specs]
    layers = [PagedLinear(pm, lr=0.15, update_groups=groups) for pm in pms]
    t0 = time.time(); steps = 0
    for ep in range(EPOCHS):
        idx = np.random.permutation(N)
        for s in range(0, N - B + 1, B):
            bi = idx[s:s+B]
            x = Tensor(X[bi]); y = Y[bi]
            h = gelu(layers[0](x)); h = gelu(layers[1](h))
            logits = layers[2](h)
            loss = cross_entropy(logits, y)
            loss.backward()
            for L in layers:
                L.tick(); L.step()
            steps += 1
    accs = []
    for s in range(0, N, 64):
        xi = Tensor(X[s:s+64])
        h = gelu(layers[0](xi)); h = gelu(layers[1](h))
        lg = layers[2](h)
        accs.append((lg.data.argmax(1) == Y[s:s+64]).mean())
    dt = time.time() - t0
    for pm in pms:
        pm.close()
    st = pool.stats()
    return float(np.mean(accs)), dt, st, steps

rows = []
for g in [1, 4]:
    acc, dt, st, steps = train(g)
    rows.append((g, acc, dt, st["hit_rate"], st["misses"], st["evictions"], steps))
    print(f"groups={g}: acc={acc:.3f}  time={dt:.1f}s  hit={st['hit_rate']*100:.1f}%  "
          f"miss={st['misses']}  evict={st['evictions']}  steps={steps}")
