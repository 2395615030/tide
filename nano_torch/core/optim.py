"""nano_torch.core.optim —— 手写优化器,不依赖任何框架。"""
import numpy as np


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd:
                p.data -= lr * self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / bc1
            vh = self.v[i] / bc2
            p.data -= lr * mh / (np.sqrt(vh) + self.eps)


class SGD:
    def __init__(self, params, lr=1e-2, momentum=0.0):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.mu = momentum
        self.buf = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            self.buf[i] = self.mu * self.buf[i] + p.grad
            p.data -= lr * self.buf[i]
