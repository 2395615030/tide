"""nano_torch.core.tensor
最小、正确、可检验的自动微分引擎。
依赖只有 numpy —— 它只当数组容器,前向/反向/优化器全部自己写,不出现任何深度学习框架。
"""
import numpy as np

F = np.float32

def _unbroadcast(grad, shape):
    """把广播过的梯度还原成 shape 的形状。"""
    if grad.shape == shape:
        return grad
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Context:
    __slots__ = ("parents", "backward")
    def __init__(self, parents, backward):
        self.parents = tuple(parents)
        self.backward = backward


class Tensor:
    __array_priority__ = 1000

    def __init__(self, data, requires_grad=False, dtype=F):
        self.data = np.asarray(data, dtype=dtype)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._ctx = None
        self._is_leaf = True

    # ---------- 基本属性 ----------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return f"Tensor(shape={self.shape}, rg={self.requires_grad})"

    def zero_grad(self):
        self.grad = None

    # ---------- 反向 ----------
    def backward(self, grad=None):
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("非标量张量必须显式给 grad")
            grad = np.ones_like(self.data)
        grad = np.asarray(grad, dtype=self.data.dtype)

        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            if t._ctx is not None:
                for p in t._ctx.parents:
                    build(p)
            topo.append(t)

        build(self)
        self.grad = grad if self.grad is None else self.grad + grad

        for t in reversed(topo):
            if t._ctx is None or t.grad is None:
                continue
            grads = t._ctx.backward(t.grad)
            for p, g in zip(t._ctx.parents, grads):
                if p is not None and p.requires_grad and g is not None:
                    g = _unbroadcast(np.asarray(g, dtype=p.data.dtype), p.shape)
                    p.grad = g if p.grad is None else p.grad + g

    # ---------- 运算符重载 ----------
    def __add__(self, o):  return add(self, _t(o))
    def __radd__(self, o): return add(_t(o), self)
    def __sub__(self, o):  return sub(self, _t(o))
    def __rsub__(self, o): return sub(_t(o), self)
    def __mul__(self, o):  return mul(self, _t(o))
    def __rmul__(self, o): return mul(_t(o), self)
    def __truediv__(self, o): return div(self, _t(o))
    def __neg__(self):     return mul(self, Tensor(-1.0))
    def __matmul__(self, o): return matmul(self, _t(o))
    def __getitem__(self, idx):
        out = self.data[idx]
        def bw(g):
            gi = np.zeros_like(self.data)
            gi[idx] = g
            return (gi,)
        return _op(out, (self,), bw)

    # ---------- 链式便利方法 ----------
    def sum(self, axis=None, keepdims=False):
        return sum_(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return mean(self, axis, keepdims)

    def reshape(self, *shape):
        return reshape(self, shape[0] if len(shape) == 1 else shape)

    def transpose(self, d0=-2, d1=-1):
        return transpose(self, d0, d1)

    def relu(self):
        return relu(self)

    def gelu(self):
        return gelu(self)

    def softmax(self, axis=-1):
        return softmax(self, axis)

    def detach(self):
        return Tensor(self.data.copy())


def _t(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def _op(data, parents, backward, rg=None):
    out = Tensor(data, dtype=np.asarray(data).dtype)
    if rg is None:
        rg = any(p.requires_grad for p in parents)
    if rg:
        out.requires_grad = True
        out._is_leaf = False
        out._ctx = Context(parents, backward)
    return out


# ---------------- 基础算子 ----------------
def add(a, b):
    return _op(a.data + b.data,
               (a, b),
               lambda g: (g, g))

def sub(a, b):
    return _op(a.data - b.data,
               (a, b),
               lambda g: (g, -g))

def mul(a, b):
    return _op(a.data * b.data,
               (a, b),
               lambda g: (g * b.data, g * a.data))

def div(a, b):
    return _op(a.data / b.data,
               (a, b),
               lambda g: (g / b.data, -g * a.data / (b.data ** 2)))

def matmul(a, b):
    ad, bd = a.data, b.data
    # [B,T,D] @ [D,F]: numpy 的 batched 路径慢 3 倍, 摊平成 2D
    if ad.ndim == 3 and bd.ndim == 2:
        B, T, D = ad.shape
        a2 = ad.reshape(B * T, D)
        bt2 = np.ascontiguousarray(bd.T)
        out = (a2 @ bd).reshape(B, T, -1)
        def bw(g):
            g2 = g.reshape(B * T, g.shape[-1])
            da = (g2 @ bt2).reshape(B, T, D)
            db = a2.T @ g2
            return da, db
        return _op(out, (a, b), bw)
    return _op(ad @ bd,
               (a, b),
               lambda g: (g @ np.swapaxes(bd, -1, -2),
                          np.swapaxes(ad, -1, -2) @ g))

def cat(tensors, axis=0):
    """沿 axis 拼接。反向按形状切回去。"""
    out = np.concatenate([t.data for t in tensors], axis=axis)
    sizes = np.cumsum([t.shape[axis] for t in tensors])[:-1]
    def bw(g):
        return tuple(np.split(g, sizes, axis=axis))
    return _op(out, tuple(tensors), bw)


def transpose(a, dim0=-2, dim1=-1):
    ax = list(range(a.ndim))
    ax[dim0], ax[dim1] = ax[dim1], ax[dim0]
    return _op(a.data.transpose(ax),
               (a,),
               lambda g: (np.transpose(g, ax),))

def reshape(a, shape):
    return _op(a.data.reshape(shape),
               (a,),
               lambda g: (g.reshape(a.shape),))

def sum_(a, axis=None, keepdims=False):
    out = a.data.sum(axis=axis, keepdims=keepdims)
    def bw(g):
        if axis is None:
            return (np.full_like(a.data, g),)
        ax = axis if isinstance(axis, (tuple, list)) else (axis,)
        g2 = g
        if not keepdims:
            for ax_i in sorted(ax):
                g2 = np.expand_dims(g2, ax_i)
        return (np.broadcast_to(g2, a.shape).copy(),)
    return _op(out, (a,), bw)

def mean(a, axis=None, keepdims=False):
    n = a.data.size if axis is None else a.shape[axis]
    return sum_(a, axis=axis, keepdims=keepdims) * (1.0 / n)


# ---------------- 常用激活 ----------------
def relu(a):
    out = np.maximum(a.data, 0)
    return _op(out, (a,), lambda g: (g * (a.data > 0),))

def gelu(a):
    """tanh 近似的 GELU,反向解析写出来。用乘法代替 x**3 (快 3 倍)。"""
    x = a.data
    c = 0.7978845608028654          # sqrt(2/pi)
    x2 = x * x
    t = np.tanh(c * (x + 0.044715 * x2 * x))
    out = 0.5 * x * (1.0 + t)
    def bw(g):
        dt = 1.0 - t * t
        dx = 0.5 * (1.0 + t) + 0.5 * x * dt * c * (1.0 + 3 * 0.044715 * x2)
        return (g * dx,)
    return _op(out, (a,), bw)

def sigmoid(a):
    s = 0.5 * (1.0 + np.tanh(0.5 * a.data))
    return _op(s, (a,), lambda g: (g * s * (1.0 - s),))


def tanh(a):
    t = np.tanh(a.data)
    return _op(t, (a,), lambda g: (g * (1.0 - t * t),))


def silu(a):
    s = 1.0 / (1.0 + np.exp(-a.data))
    out = a.data * s
    return _op(out, (a,), lambda g: (g * (s + a.data * s * (1 - s)),))


# ---------------- 归一化 / 注意力底座 ----------------
class LayerNormFn:
    @staticmethod
    def forward(x, gamma, beta, eps=1e-5):
        mu = x.data.mean(-1, keepdims=True)
        xc = x.data - mu
        var = (xc ** 2).mean(-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + eps)
        xh = xc * rstd
        out = xh * gamma.data + beta.data
        def bw(g):
            N = x.shape[-1]
            dgamma = (g * xh).reshape(-1, N).sum(0).reshape(gamma.shape)
            dbeta = g.reshape(-1, N).sum(0).reshape(beta.shape)
            dxh = g * gamma.data
            dx = (rstd / N) * (N * dxh
                               - dxh.sum(-1, keepdims=True)
                               - xh * (dxh * xh).sum(-1, keepdims=True))
            return dx, dgamma, dbeta
        return _op(out, (x, gamma, beta), bw)


def layernorm(x, gamma, beta, eps=1e-5):
    return LayerNormFn.forward(x, gamma, beta, eps)


def softmax(x, axis=-1):
    m = x.data.max(axis=axis, keepdims=True)
    e = np.exp(x.data - m)
    s = e / e.sum(axis=axis, keepdims=True)
    def bw(g):
        return (s * (g - (g * s).sum(axis=axis, keepdims=True)),)
    return _op(s, (x,), bw)


# ---------------- 损失 ----------------
def cross_entropy(logits, targets):
    """logits: [..., V]  targets: [...] int64  -> 标量 loss

    不构造完整的 log-prob 张量, 只取目标位置的 logit 与 logsumexp。
    """
    x = logits.data
    n = targets.size
    flat = x.reshape(n, -1)
    tgt = np.asarray(targets).reshape(-1)
    idx = np.arange(n)
    m = flat.max(-1, keepdims=True)
    z = np.exp(flat - m)
    s = z.sum(-1, keepdims=True)
    loss = -(flat[idx, tgt] - (m + np.log(s)).reshape(n)).mean()

    def bw(g):
        p = z                         # 复用已经算好的 exp(flat-m)
        p /= s                        # softmax
        p[idx, tgt] -= 1.0
        p /= n
        return (g * p.reshape(logits.shape),)
    return _op(loss, (logits,), bw)


def embedding(weight, idx):
    idx = np.asarray(idx, dtype=np.int64)
    out = weight.data[idx]
    def bw(g):
        dw = np.zeros_like(weight.data)
        np.add.at(dw, idx.reshape(-1), g.reshape(-1, weight.shape[-1]))
        return (dw,)
    return _op(out, (weight,), bw)


# ---------------- 数值梯度检验 ----------------
def numeric_grad(fn, t, eps=1e-6):
    """fn: () -> Tensor(标量)。返回 t 的数值梯度（直接扰动 t.data）。"""
    arr = t.data
    backup = arr.copy()
    flat = arr.reshape(-1)
    g = np.zeros(flat.size, dtype=np.float64)
    try:
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps
            lp = float(fn().data)
            flat[i] = old - eps
            lm = float(fn().data)
            flat[i] = old
            g[i] = (lp - lm) / (2 * eps)
    finally:
        arr[...] = backup
    return g.reshape(arr.shape)
