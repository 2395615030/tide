"""命中率 vs 池容量: 判断瓶颈是算法还是容量。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from nano_torch.core.store import PagedMatrix, CachePool

D = "D:/ROOT/Projects/DSH/Def/nano-torch/data"
path = os.path.join(D, "W2.ndst")   # 1024x1024 fp32 = 4 MB, 16 chunks x 256 KB

X = np.random.randn(32, 1024).astype(np.float32)
REPEAT = 10

print(f"{'pool':>10} {'chunks':>7} {'hit%':>7} {'miss':>6} {'evict':>6} {'time':>8}")
print("-" * 50)
for mb in [0.25, 0.5, 1, 2, 3, 4, 5]:
    cap = int(mb * 1024 * 1024)
    pool = CachePool(cap)
    pm = PagedMatrix(path, pool)
    # 预热一次, 然后重复读同样的块 (模拟跨 step 复用)
    t0 = time.time()
    for _ in range(REPEAT):
        pm.dot_t(X)
        pm.pool.unpin(pm._key(0))
    dt = time.time() - t0
    s = pool.stats()
    print(f"{mb:>8.2f}M {pool.capacity//pm.chunk_bytes:>7} {s['hit_rate']*100:>6.1f}% "
          f"{s['misses']:>6} {s['evictions']:>6} {dt*1000:>7.0f}ms")
    pm.close()
print("\n注: 每次循环读的是同一份 W, 所以越大的池命中率越高。")
print(f"W2 = 1024x1024 fp32 = 4 MB = 16 chunks x 256 KB, 读 {REPEAT} 次")
