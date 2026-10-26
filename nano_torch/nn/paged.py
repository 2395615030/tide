"""nano_torch.nn.paged —— 分块参数层。

fused=True 时, 反向与更新在同一次块遍历里完成:
  读块 -> 算 dW -> 算 dx -> 更新 -> 写回
每块每步只碰两次(前进一次 + 后退一次), 而不是三四次。
"""
import numpy as np

from ..core.tensor import Tensor, _op
from ..core.store import PagedMatrix, CachePool


class PagedLinear:
    def __init__(self, pm: PagedMatrix, bias=True, lr=0.05, fused=True, update_groups=1):
        self.pm = pm
        self.lr = lr
        self.fused = fused
        self.bias = (Tensor(np.zeros(pm.rows, dtype=np.float32), requires_grad=True)
                     if bias else None)
        self._x = None
        self._dW = None
        self.updated = 0
        self.update_groups = max(1, int(update_groups))
        self._step = 0

    def __call__(self, x: Tensor) -> Tensor:
        y = self.pm.dot_t(x.data)
        if self.bias is not None:
            y = y + self.bias.data
        self._x = x.data

        def bw(g):
            g = np.asarray(g, dtype=np.float32)
            dx = np.zeros((g.shape[0], self.pm.cols), dtype=np.float32)
            xin = self._x
            if not self.fused:
                self._dW = {}
            for i in range(self.pm.n_chunks):
                r0 = i * self.pm.chunk_rows
                r1 = min(r0 + self.pm.chunk_rows, self.pm.rows)
                key = self.pm._key(i)
                w = self.pm.chunk(i)[: r1 - r0]        # 读一次
                gi = g[:, r0:r1]
                dx += gi @ w                            # 反向用
                if not self.fused:
                    self._dW[i] = self.pm.grad_chunk(g, xin, i)
                elif (i % self.update_groups) == (self._step % self.update_groups):
                    dw = gi.T @ xin                     # 梯度
                    new_w = self.pm.chunk(i).copy()
                    new_w[: r1 - r0] -= self.lr * dw
                    self.pm.write_chunk(i, new_w)       # 立即写回
                    self.pm.pool.invalidate(key)
                    self.updated += 1
                self.pm.pool.unpin(key)
            if self.bias is None:
                return (dx,)
            self._db = g.sum(0).astype(np.float32)
            return (dx, self._db)

        parents = (x,) if self.bias is None else (x, self.bias)
        return _op(y.astype(np.float32), parents, bw)

    def tick(self):
        self._step += 1

    def step(self):
        if self.fused:
            if self.bias is not None and self.bias.grad is not None:
                self.bias.data -= self.lr * self.bias.grad
                self.bias.grad = None
            return
        if self._dW is None:
            return
        for i, g in self._dW.items():
            w = self.pm.chunk(i).copy()
            w -= self.lr * np.asarray(g, dtype=np.float32)
            self.pm.write_chunk(i, w)
            self.pm.pool.invalidate(self.pm._key(i))
        self._dW = None
        if self.bias is not None and self.bias.grad is not None:
            self.bias.data -= self.lr * self.bias.grad
            self.bias.grad = None

    def zero_grad(self):
        self._dW = None
        if self.bias is not None:
            self.bias.grad = None
