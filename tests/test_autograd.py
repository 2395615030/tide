import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from nano_torch.core.tensor import (
    Tensor, add, sub, mul, div, matmul, transpose, reshape, sum_, mean,
    relu, gelu, silu, layernorm, softmax, cross_entropy, embedding, numeric_grad,
)

np.random.seed(0)
D = np.float64

def T(*shape, scale=1.0):
    return Tensor(np.random.randn(*shape).astype(D) * scale, requires_grad=True, dtype=D)

def run(name, tlist, fwd, tol=2e-6):
    for t in tlist:
        t.zero_grad()
    out = fwd()
    out.backward()
    worst = 0.0
    for t in tlist:
        num = numeric_grad(fwd, t)
        ana = t.grad
        denom = np.max(np.abs(num)) + 1e-12
        err = np.max(np.abs(num - ana)) / denom
        worst = max(worst, err)
    flag = "OK  " if worst < tol else "FAIL"
    print(f"{name:24s} {flag} relerr={worst:.3e}")

# ---- 单算子 ----
a = T(3, 4); b = T(3, 4)
run("add", [a, b], lambda: (a + b).sum())

a = T(3, 4); b = T(4, 5)
run("matmul", [a, b], lambda: (a @ b).sum())

a = T(3, 4); b = T(3, 4)
run("mul", [a, b], lambda: (a * b).sum())

a = T(3, 4); b = T(3, 4)
run("div", [a, b], lambda: (a / b).sum())

x = T(2, 3, 4)
run("transpose", [x], lambda: transpose(x, 1, 2).sum())

x = T(5, 6)
run("relu", [x], lambda: relu(x).sum())

x = T(5, 6)
run("gelu", [x], lambda: gelu(x).sum())

x = T(5, 6)
run("silu", [x], lambda: silu(x).sum())

x = T(4, 8); g = T(8); be = T(8)
run("layernorm", [x, g, be], lambda: layernorm(x, g, be).sum())

x = T(4, 8)
run("softmax", [x], lambda: (softmax(x) * x).sum())

lg = T(6, 10)
tgt = np.random.randint(0, 10, size=6)
run("cross_entropy", [lg], lambda: cross_entropy(lg, tgt))

W = T(16, 8)
idx = np.random.randint(0, 16, size=(3, 5))
run("embedding", [W], lambda: embedding(W, idx).sum())

# ---- 组合: 两层 MLP + CE ----
x = T(8, 6)
W1 = T(6, 12); b1 = T(1, 12)
W2 = T(12, 5); b2 = T(1, 5)
y = np.random.randint(0, 5, size=8)
def fwd():
    h = gelu(x @ W1 + b1)
    logits = h @ W2 + b2
    return cross_entropy(logits, y)
run("2layer MLP + CE", [x, W1, b1, W2, b2], fwd)

print("\n--- gradcheck done ---")
