"""nano_torch.core.store —— 分块权重存储 + 容量受限缓存池（NEFS 雏形）。

设计目标:
  L0: 磁盘上的分块文件 (mmap 只读)
  L1: 容量固定的内存池, 按 LRU 淘汰, pin 住的不许淘汰
  L2: (留给 GPU, 本版不做)

关键点: 任何时刻 L1 的驻留字节数 <= capacity, 这是"极限内存管理"的核心不变量。
"""
import mmap
import os
import struct
from collections import OrderedDict, defaultdict

import numpy as np

MAGIC = b"NDST"
VERSION = 1
DTYPE_F32 = 0
HEADER_SIZE = 4096          # 对齐到页, 方便以后加 chunk table

_HDR = struct.Struct("<4sIIIIIQ")   # magic, version, rows, cols, chunk_rows, dtype, data_off


def save_paged(path, arr, chunk_rows=64):
    """把一个 2D 矩阵按行分块写成文件。"""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    rows, cols = arr.shape
    n_chunks = (rows + chunk_rows - 1) // chunk_rows
    with open(path, "wb") as f:
        hdr = _HDR.pack(MAGIC, VERSION, rows, cols, chunk_rows, DTYPE_F32, HEADER_SIZE)
        f.write(hdr.ljust(HEADER_SIZE, b"\0"))
        for i in range(n_chunks):
            r0 = i * chunk_rows
            r1 = min(r0 + chunk_rows, rows)
            chunk = np.zeros((chunk_rows, cols), dtype=np.float32)
            chunk[: r1 - r0] = arr[r0:r1]
            f.write(chunk.tobytes())
    return n_chunks


class CachePool:
    """固定容量的缓存池。真实地持有 bytes, 超容量就淘汰。"""

    def __init__(self, capacity_bytes):
        self.capacity = int(capacity_bytes)
        self.resident = 0
        self.peak = 0
        self._data = OrderedDict()      # key -> np.ndarray (bytes)
        self._pins = defaultdict(int)
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def available(self):
        return self.capacity - self.resident

    def get(self, key, loader):
        if key in self._data:
            self.hits += 1
            self._data.move_to_end(key)
            return self._data[key]
        self.misses += 1
        block = loader()                 # 从 L0 读进来
        nbytes = block.nbytes
        if nbytes > self.capacity:
            raise MemoryError(f"单块 {nbytes} 字节超过池容量 {self.capacity}")
        while self.resident + nbytes > self.capacity:
            if not self._evict_one(exclude=key):
                raise MemoryError("池已满且全部被 pin 住")
        self._data[key] = block
        self.resident += nbytes
        self.peak = max(self.peak, self.resident)
        return block

    def _evict_one(self, exclude=None):
        for k in list(self._data.keys()):
            if k == exclude:
                continue
            if self._pins.get(k, 0) > 0:
                continue
            block = self._data.pop(k)
            self.resident -= block.nbytes
            self.evictions += 1
            return True
        return False

    def invalidate(self, key):
        if key in self._data:
            self.resident -= self._data[key].nbytes
            self._data.pop(key)

    def pin(self, key):
        self._pins[key] += 1

    def unpin(self, key):
        if self._pins.get(key, 0) > 0:
            self._pins[key] -= 1

    def stats(self):
        return {
            "capacity": self.capacity,
            "resident": self.resident,
            "peak": self.peak,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hits / max(1, self.hits + self.misses),
        }


class PagedMatrix:
    """逻辑上是 [rows, cols] 的矩阵, 物理上按行分块躺在文件里。"""

    def __init__(self, path, pool=None, writable=False):
        self.path = path
        self.writable = writable
        self.f = open(path, "r+b" if writable else "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0,
                            access=mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ)
        magic, version, rows, cols, chunk_rows, dtype, data_off = _HDR.unpack_from(self.mm, 0)
        if magic != MAGIC:
            raise ValueError(f"不是 NDST 文件: {path}")
        if version != VERSION:
            raise ValueError(f"版本不支持: {version}")
        if dtype != DTYPE_F32:
            raise ValueError("本版只支持 float32")
        self.rows, self.cols = rows, cols
        self.chunk_rows = chunk_rows
        self.data_off = data_off
        self.n_chunks = (rows + chunk_rows - 1) // chunk_rows
        self.chunk_bytes = chunk_rows * cols * 4
        self.pool = pool if pool is not None else CachePool(self.chunk_bytes * 2)
        self.file_size = os.path.getsize(path)

    def _key(self, i):
        return (self.path, i)

    def _load(self, i):
        off = self.data_off + i * self.chunk_bytes
        raw = self.mm[off: off + self.chunk_bytes]
        return np.frombuffer(raw, dtype=np.float32).copy()

    def chunk(self, i, pin=False):
        if pin:
            self.pool.pin(self._key(i))
        block = self.pool.get(self._key(i), lambda: self._load(i))
        return block.reshape(self.chunk_rows, self.cols)

    def release(self, i):
        self.pool.unpin(self._key(i))

    # ---- 核心: y = x @ W.T, 逐块流式, 不用把 W 整体读进来 ----
    def dot_t(self, x, pin_chunks=False):
        x = np.ascontiguousarray(x, dtype=np.float32)
        rows = x.shape[0]
        out = np.empty((rows, self.rows), dtype=np.float32)
        n = self.n_chunks
        for i in range(n):
            r0 = i * self.chunk_rows
            r1 = min(r0 + self.chunk_rows, self.rows)
            if pin_chunks:
                self.pool.pin(self._key(i))
            w = self.chunk(i).reshape(self.chunk_rows, self.cols)
            out[:, r0:r1] = x @ w[: r1 - r0].T
            if pin_chunks:
                self.pool.unpin(self._key(i))
            else:
                self.pool.unpin(self._key(i))   # chunk() 未 pin, 这里无副作用
        return out

    def dot(self, y):
        """dx = y @ W,  y: [B, rows] -> [B, cols]。反向用, 同样逐块。"""
        y = np.ascontiguousarray(y, dtype=np.float32)
        dx = np.zeros((y.shape[0], self.cols), dtype=np.float32)
        for i in range(self.n_chunks):
            r0 = i * self.chunk_rows
            r1 = min(r0 + self.chunk_rows, self.rows)
            w = self.chunk(i)
            dx += y[:, r0:r1] @ w[: r1 - r0]
            self.pool.unpin(self._key(i))
        return dx

    def grad_chunk(self, y, x, i):
        """dW[i] = y[:, r0:r1].T @ x,  y: [B, rows], x: [B, cols]"""
        r0 = i * self.chunk_rows
        r1 = min(r0 + self.chunk_rows, self.rows)
        g = np.zeros((self.chunk_rows, self.cols), dtype=np.float32)
        g[: r1 - r0] = y[:, r0:r1].T @ x
        return g

    def write_chunk(self, i, block):
        if not self.writable:
            raise PermissionError("只读打开, 不能写回")
        off = self.data_off + i * self.chunk_bytes
        self.mm[off: off + self.chunk_bytes] = np.ascontiguousarray(
            block, dtype=np.float32).tobytes()

    def dense(self):
        """仅供校验用: 把整块读进来。会撑爆池, 慎用。"""
        out = np.empty((self.rows, self.cols), dtype=np.float32)
        for i in range(self.n_chunks):
            r0 = i * self.chunk_rows
            r1 = min(r0 + self.chunk_rows, self.rows)
            out[r0:r1] = self._load(i)[: r1 - r0].reshape(r1 - r0, self.cols)
        return out

    def close(self):
        try:
            self.mm.close()
        finally:
            self.f.close()
