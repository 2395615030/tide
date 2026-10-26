import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from nano_torch.core.store import save_paged, PagedMatrix, CachePool

np.random.seed(7)
D = "D:/ROOT/Projects/DSH/Def/nano-torch/data"
os.makedirs(D, exist_ok=True)
path = os.path.join(D, "W_test.ndst")

# ---- 1. 造一个大矩阵: 4096 x 2048 fp32 = 32 MB ----
ROWS, COLS, CHUNK_ROWS = 4096, 2048, 64
W = np.random.randn(ROWS, COLS).astype(np.float32)
print(f"W: {ROWS}x{COLS} = {W.nbytes/1024/1024:.1f} MB, chunk_rows={CHUNK_ROWS}")
print(f"chunk size = {CHUNK_ROWS*COLS*4/1024:.0f} KB -> {(ROWS+CHUNK_ROWS-1)//CHUNK_ROWS} chunks")

t0 = time.time()
n = save_paged(path, W, chunk_rows=CHUNK_ROWS)
print(f"saved: {n} chunks, file = {os.path.getsize(path)/1024/1024:.1f} MB, {time.time()-t0:.2f}s")

# ---- 2. 用 1 MB 的池跑 32 MB 的矩阵 ----
POOL = 1024 * 1024
pool = CachePool(POOL)
pm = PagedMatrix(path, pool)
print(f"\npool capacity = {POOL/1024:.0f} KB ({POOL // pm.chunk_bytes} chunks)")

X = np.random.randn(32, COLS).astype(np.float32)

t0 = time.time()
Y_paged = pm.dot_t(X)
t_paged = time.time() - t0

t0 = time.time()
Y_ref = X @ W.T
t_dense = time.time() - t0

err = np.max(np.abs(Y_paged - Y_ref))
print(f"\npaged  : {t_paged*1000:.1f} ms")
print(f"dense  : {t_dense*1000:.1f} ms   ({t_paged/t_dense:.1f}x slower)")
print(f"max err: {err:.2e}  -> {'OK' if err < 1e-4 else 'FAIL'}")

s = pool.stats()
print("\n--- pool stats ---")
for k, v in s.items():
    print(f"  {k:10s}: {v:.4f}" if isinstance(v, float) else f"  {k:10s}: {v}")
print(f"\nresident peak = {s['peak']/1024:.0f} KB  (W 本体 {W.nbytes/1024:.0f} KB)")
print(f"压缩比: {W.nbytes / max(1, s['peak']):.1f}x")

# ---- 3. 极端: 池只放 1 块 ----
pool2 = CachePool(pm.chunk_bytes)
pm2 = PagedMatrix(path, pool2)
Y2 = pm2.dot_t(X)
print(f"\n单块池 ({pm2.chunk_bytes/1024:.0f} KB): err={np.max(np.abs(Y2-Y_ref)):.2e}, evictions={pool2.evictions}")

pm.close(); pm2.close()
