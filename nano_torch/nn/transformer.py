"""nano_torch.nn.transformer —— 极简 decoder-only Transformer + 情绪调制。"""
import numpy as np

from ..core.tensor import (
    Tensor, matmul, transpose, reshape, mean, layernorm, softmax, gelu, tanh, relu, silu,
)


def param(*shape, scale=None):
    if scale is None:
        scale = np.sqrt(2.0 / max(1, shape[0]))
    return Tensor((np.random.randn(*shape) * scale).astype(np.float32), requires_grad=True)


def mha(x, Wq, Wk, Wv, Wo, n_heads, mask):
    B, T, C = x.shape
    D = C // n_heads
    q = reshape(x @ Wq, (B, T, n_heads, D)).transpose(1, 2)
    k = reshape(x @ Wk, (B, T, n_heads, D)).transpose(1, 2)
    v = reshape(x @ Wv, (B, T, n_heads, D)).transpose(1, 2)
    att = softmax((q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(D)) + mask, axis=-1)
    out = att @ v
    out = reshape(out.transpose(1, 2), (B, T, C))
    return out @ Wo


class Block:
    def __init__(self, d, h, ff, act='gelu', use_state=False):
        self.nh = h
        self.act = {'gelu': gelu, 'relu': relu, 'silu': silu}[act]
        self.use_state = use_state
        if use_state:
            self.Ws = param(d, d, scale=0.02)
        self.Wq, self.Wk, self.Wv, self.Wo = (param(d, d) for _ in range(4))
        self.g1, self.b1 = param(d, scale=1.0), param(d, scale=0.01)
        self.g2, self.b2 = param(d, scale=1.0), param(d, scale=0.01)
        self.W1, self.b1f = param(d, ff), param(ff, scale=0.01)
        self.W2, self.b2f = param(ff, d), param(d, scale=0.01)

    def params(self):
        ps = [self.Wq, self.Wk, self.Wv, self.Wo, self.g1, self.b1,
                self.g2, self.b2, self.W1, self.b1f, self.W2, self.b2f]
        if self.use_state:
            ps.append(self.Ws)
        return ps

    def __call__(self, x, mask, state=None):
        h = layernorm(x, self.g1, self.b1)
        x = x + mha(h, self.Wq, self.Wk, self.Wv, self.Wo, self.nh, mask)
        h2 = layernorm(x, self.g2, self.b2)
        f = self.act(h2 @ self.W1 + self.b1f) @ self.W2 + self.b2f
        if self.use_state and state is not None:
            x = x + (state @ self.Ws).reshape(x.shape[0], 1, -1)
        return x + f


class TinyLM:
    def __init__(self, vocab, d=128, n_heads=4, n_layers=2, ff=512, max_T=128,
                 use_emotion=False, emotion_dim=64, gamma=1.0, act='gelu', use_state=False):
        self.vocab = vocab
        self.max_T = max_T
        self.use_emotion = use_emotion
        self.gamma = gamma
        self.tok = param(vocab, d)
        self.pos = param(max_T, d, scale=0.01)
        self.blocks = [Block(d, n_heads, ff, act, use_state) for _ in range(n_layers)]
        self.use_state = use_state
        self.gf, self.bf = param(d, scale=1.0), param(d, scale=0.01)
        self.head = param(d, vocab)
        if use_state:
            self.dom = param(8, d, scale=0.02)
            self.Wh = param(d, d, scale=0.02)
            self.Wx = param(d, d, scale=0.02)
        if use_emotion:
            self.We_in = param(d, emotion_dim, scale=0.05)
            self.We_out = param(emotion_dim, vocab, scale=0.05)

    def params(self):
        ps = [self.tok, self.pos, self.gf, self.bf, self.head]
        for b in self.blocks:
            ps += b.params()
        if self.use_emotion:
            ps += [self.We_in, self.We_out]
        if self.use_state:
            ps += [self.dom, self.Wh, self.Wx]
        return ps

    def __call__(self, idx, mask, dom=None, h=None, return_state=False):
        idx = np.asarray(idx.data if isinstance(idx, Tensor) else idx, dtype=np.int64)
        B, T = idx.shape
        state = h
        if state is None and self.use_state and dom is not None:
            state = self.dom[np.asarray(dom, dtype=np.int64)]
        x = self.tok[idx] + self.pos[np.arange(T)].reshape(1, T, -1)
        for b in self.blocks:
            x = b(x, mask, state)
        x = layernorm(x, self.gf, self.bf)
        if self.use_emotion:
            pooled = mean(x, axis=1)
            e = tanh(pooled @ self.We_in)
            bias = reshape(e @ self.We_out, (B, 1, -1))
            logits = x @ self.head + bias * self.gamma
        else:
            logits = x @ self.head
        if return_state and self.use_state:
            pooled = mean(x, axis=1)
            if state is None:
                h_new = tanh(pooled @ self.Wx)
            else:
                h_new = tanh(state @ self.Wh + pooled @ self.Wx)
            return logits, h_new
        return logits
