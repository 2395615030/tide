"""低资源训练验证: 5 MB 权重, 512 KB 缓存池, 跑完整的前向/反向/更新。"""
import sys, os, time, tracemalloc
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
import numpy as np
from nano_torch.core.tensor import Tensor, gelu, cross_entropy
from nano_torch.core.store import save_paged, PagedMatrix, CachePool
from nano_torch.nn.paged import PagedLinear

np.random.seed(0)
D = "D:/ROOT/Projects/DSH/Def/nano-torch/data"
IN, H, OUT, CHUNK = 256, 1024, 16, 64
POOL = 512 * 1024

# ---- 造三层权重文件 ----
specs = [("W1", (H, IN), 0.05), ("W2", (H, H), 0.02), ("W3", (OUT, H), 0.02)]
total = 0
for name, shape, s in specs:
    W = (np.random.randn(*shape) * s).astype(np.float32)
    total += W.nbytes
    save_paged(os.path.join(D, name + ".ndst"), W, CHUNK)
    print(f"{name}: {shape} = {W.nbytes/1024:.0f} KB")
print(f"总权重: {total/1024/1024:.2f} MB | 池: {POOL/1024:.0f} KB | 块: {CHUNK}"

      )

# ---- 模型 ----
pool = CachePool(POOL)
pms = [PagedMatrix(os.path.join(D, n + ".ndst"), pool, writable=True) for n, _, _ in specs]
layers = [PagedLinear(pm, lr=0.15) for pm in pms]

# ---- 数据: 标签 = 前 16 维里最大的那个 ----
N, B = 512, 32
X = np.random.randn(N, IN).astype(np.float32)
Y = X[:, :OUT].argmax(1).astype(np.int64)

tracemalloc.start()
t0 = time.time()
step = 0
for epoch in range(18):
    idx = np.random.permutation(N)
    for s in range(0, N - B + 1, B):
        bi = idx[s:s + B]
        x = Tensor(X[bi])
        y = Y[bi]
        h = gelu(layers[0](x))
        h = gelu(layers[1](h))
        logits = layers[2](h)
        loss = cross_entropy(logits, y)
        loss.backward()
        for L in layers:
            L.step()
        step += 1
        if step % 40 == 0:
            cur, peak = tracemalloc.get_traced_memory()
            acc = (logits.data.argmax(1) == y).mean()
            print(f"step {step:4d}  loss {float(loss.data):.4f}  acc {acc:.2f}  "
                  f"pool {pool.resident/1024:6.0f} KB (peak {pool.peak/1024:6.0f})  "
                  f"tracemalloc {cur/1024/1024:.1f} MB")
    # 一个 epoch 结束, 测全量准确率
    accs = []
    for s in range(0, N, 64):
        xi = Tensor(X[s:s+64])
        h = gelu(layers[0](xi)); h = gelu(layers[1](h))
        lg = layers[2](h)
        accs.append((lg.data.argmax(1) == Y[s:s+64]).mean())
    print(f"  epoch {epoch}: train acc {np.mean(accs):.3f}")

cur, peak = tracemalloc.get_traced_memory()
print(f"\n--- 结果 ---")
print(f"总权重        : {total/1024/1024:.2f} MB")
print(f"缓存池容量    : {POOL/1024:.0f} KB")
print(f"池峰值驻留    : {pool.peak/1024:.0f} KB")
print(f"Python 峰值   : {peak/1024/1024:.1f} MB")
print(f"压缩比        : {total/max(pool.peak,1):.1f}x")
s = pool.stats()
print(f"命中率        : {s['hit_rate']*100:.1f}%  ({s['hits']} hit / {s['misses']} miss)")
print(f"淘汰次数      : {s['evictions']}")
print(f"总步数        : {step}  用时 {time.time()-t0:.1f}s")
