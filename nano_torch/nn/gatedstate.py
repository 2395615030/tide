"""带内容门控的多尺度状态层。"""
import numpy as np

from ..core.tensor import Tensor, matmul, cat, sigmoid, sum_

from .statelm import MultiScaleState, param


class GatedState(MultiScaleState):
    """在纯时间衰减之上, 加一个低秩的内容门: 决定这一刻往状态里写多少。

        g_t = sigmoid( q_t · mean(k_{<=t}) )
        h   = a*h + (1-a) * g_t * v_t
    """
    def __init__(self, d, taus=(2, 8, 32, 128), gate_dim=8):
        super().__init__(d, taus)
        self.gate_dim = gate_dim
        self.Wq = param(d, gate_dim, scale=0.05)
        self.Wk = param(d, gate_dim, scale=0.05)
        self._C = {}

    def _C_for(self, T):
        if T in self._C:
            return self._C[T]
        i = np.arange(T)[:, None]
        j = np.arange(T)[None, :]
        C = np.where(i >= j, 1.0 / (i + 1.0), 0.0).astype(np.float32)
        self._C[T] = C
        return C

    def params(self):
        return super().params() + [self.Wq, self.Wk]

    def __call__(self, x, emo_bias=None):
        B, T, D = x.shape
        V = x @ self.W_in
        q = x @ self.Wq
        k = x @ self.Wk
        ctx = matmul(Tensor(self._C_for(T)), k)          # [B,T,G]
        pre = q * ctx
        if emo_bias is not None:
            pre = pre + emo_bias
        g = sigmoid(sum_(pre, axis=-1, keepdims=True))  # [B,T,1]
        Vg = V * g
        A = self._A_for(T)
        hs = [matmul(Tensor(A[s]), Vg) for s in range(self.S)]
        H = cat(hs, axis=2)
        y = H @ self.W_mix
        return x + y @ self.W_out
