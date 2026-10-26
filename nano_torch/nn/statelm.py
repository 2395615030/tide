"""多尺度状态层: 用因果卷积矩阵并行计算线性递归, 替代 attention。"""
import numpy as np

from ..core.tensor import Tensor, matmul, layernorm, cat, gelu


def param(*shape, scale=None):
    if scale is None:
        scale = np.sqrt(2.0 / max(1, shape[0]))
    return Tensor((np.random.randn(*shape) * scale).astype(np.float32), requires_grad=True)


class MultiScaleState:
    """h_s(t) = a_s * h_s(t-1) + (1-a_s) * v(t),  s 是不同的时间尺度。

    用下三角矩阵 A_s 一次性算出整段序列:
        H_s = A_s @ V,   A_s[t,k] = a_s^(t-k) * (1-a_s)  (k<=t)
    这样没有逐 token 循环, 训练速度可控。
    """
    def __init__(self, d, taus=(2, 8, 32, 128)):
        self.taus = taus
        self.S = len(taus)
        self.a = np.array([np.exp(-1.0 / t) for t in taus], dtype=np.float32)
        self.W_in = param(d, d)
        self.W_mix = param(self.S * d, d)
        self.W_out = param(d, d)
        self._A = {}

    def _A_for(self, T):
        if T in self._A:
            return self._A[T]
        A = np.zeros((self.S, T, T), dtype=np.float32)
        idx = np.arange(T)
        for s, a in enumerate(self.a):
            k = idx[:, None] - idx[None, :]          # t - k
            mask = k >= 0
            A[s] = np.where(mask, (a ** np.clip(k, 0, None)) * (1.0 - a), 0.0)
        self._A[T] = A
        return A

    def params(self):
        return [self.W_in, self.W_mix, self.W_out]

    def __call__(self, x, emo_bias=None):
        B, T, D = x.shape
        V = x @ self.W_in
        A = self._A_for(T)
        hs = [matmul(Tensor(A[s]), V) for s in range(self.S)]
        H = cat(hs, axis=2)                          # [B,T,S*D]
        y = H @ self.W_mix
        return x + y @ self.W_out


class StateBlock:
    def __init__(self, d, taus=(2, 8, 32, 128), ff=None, gated=False):
        ff = ff or 4 * d
        if gated:
            from .gatedstate import GatedState
            self.mix = GatedState(d, taus)
        else:
            self.mix = MultiScaleState(d, taus)
        self.g1, self.b1 = param(d, scale=1.0), param(d, scale=0.01)
        self.g2, self.b2 = param(d, scale=1.0), param(d, scale=0.01)
        self.W1, self.b1f = param(d, ff), param(ff, scale=0.01)
        self.W2, self.b2f = param(ff, d), param(d, scale=0.01)

    def params(self):
        return (self.mix.params() +
                [self.g1, self.b1, self.g2, self.b2, self.W1, self.b1f, self.W2, self.b2f])

    def __call__(self, x, emo_bias=None):
        h = layernorm(x, self.g1, self.b1)
        x = self.mix(h, emo_bias)
        h2 = layernorm(x, self.g2, self.b2)
        return x + gelu(h2 @ self.W1 + self.b1f) @ self.W2 + self.b2f


class StateLM:
    """完全不用 attention 的字符级模型。"""
    def __init__(self, vocab, d=128, n_layers=3, max_T=128, taus=(2, 8, 32, 128), gated=False):
        self.vocab = vocab
        self.max_T = max_T
        self.tok = param(vocab, d)
        self.pos = param(max_T, d, scale=0.01)
        self.blocks = [StateBlock(d, taus, gated=gated) for _ in range(n_layers)]
        self.gf, self.bf = param(d, scale=1.0), param(d, scale=0.01)
        self.head = param(d, vocab)
        self.W_emo = param(4, 1, scale=0.1)

    def params(self):
        ps = [self.tok, self.pos, self.gf, self.bf, self.head, self.W_emo]
        for b in self.blocks:
            ps += b.params()
        return ps

    def __call__(self, idx, mask=None, emo=None):
        idx = np.asarray(idx.data if isinstance(idx, Tensor) else idx, dtype=np.int64)
        B, T = idx.shape
        x = self.tok[idx] + self.pos[np.arange(T)].reshape(1, T, -1)
        eb = None
        if emo is not None:
            eb = float(np.asarray(emo, np.float64) @ self.W_emo.data.astype(np.float64).ravel())
        for b in self.blocks:
            x = b(x, eb)
        x = layernorm(x, self.gf, self.bf)
        return x @ self.head
