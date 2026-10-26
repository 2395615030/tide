import sys, os, time
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
import numpy as np
from nano_torch.core.tensor import Tensor, gelu, sub, mul, mean
from nano_torch.core.optim import AdamW

np.random.seed(42)
F = np.float32

def param(*shape, scale=None):
    if scale is None:
        scale = np.sqrt(2.0 / shape[0])
    return Tensor((np.random.randn(*shape) * scale).astype(F), requires_grad=True)

def linear(x, W, b):
    return x @ W + b

# ---- 数据: 拟合 sin(x) ----
N = 512
X = np.linspace(-np.pi, np.pi, N).reshape(-1, 1).astype(F)
Y = np.sin(X).astype(F)

W1, b1 = param(1, 64), param(1, 64, scale=0.1)
W2, b2 = param(64, 64), param(1, 64, scale=0.1)
W3, b3 = param(64, 1), param(1, 1, scale=0.1)
params = [W1, b1, W2, b2, W3, b3]
opt = AdamW(params, lr=3e-3, weight_decay=1e-4)

xt = Tensor(X)
yt = Tensor(Y)

def mse(pred, target):
    d = sub(pred, target)
    return mean(mul(d, d))

t0 = time.time()
for step in range(1500):
    pred = linear(gelu(linear(gelu(linear(xt, W1, b1)), W2, b2)), W3, b3)
    loss = mse(pred, yt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 150 == 0 or step == 1499:
        print(f"step {step:5d}  loss {float(loss.data):.6f}")

pred = linear(gelu(linear(gelu(linear(xt, W1, b1)), W2, b2)), W3, b3)
err = np.abs(pred.data - Y)
print(f"\nfinal loss   : {float(mse(pred, yt).data):.6f}")
print(f"max abs err  : {err.max():.4f}")
print(f"mean abs err : {err.mean():.4f}")
print(f"train time   : {time.time()-t0:.2f}s  ({1500} steps)")
print(f"params       : {sum(p.size for p in params)}")
