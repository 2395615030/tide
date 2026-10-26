"""加载 checkpoint, 采样生成。"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
import numpy as np
from nano_torch.core.tensor import Tensor
from nano_torch.nn.transformer import TinyLM
from nano_torch.data.neko import load_neko

text, ids, keep = load_neko()
V = len(keep) + 1
T = 96
rev = {i + 1: ch for i, ch in enumerate(keep)}
rev[0] = "?"
stoi = {ch: i + 1 for i, ch in enumerate(keep)}

ck = np.load("D:/ROOT/Projects/DSH/Def/nano-torch/ckpt/neko_d128_l3.npz")
model = TinyLM(V, d=128, n_heads=4, n_layers=3, ff=512, max_T=T, act="silu")
for i, p in enumerate(model.params()):
    p.data = ck[f"p{i}"].astype(np.float32)

def generate(prompt, n=160, temp=0.8, seed=0):
    rng = np.random.RandomState(seed)
    out = [stoi.get(c, 0) for c in prompt]
    with np.errstate(all="ignore"):
        for _ in range(n):
            ctx = out[-T:]
            L = len(ctx)
            m = np.triu(np.full((L, L), -1e9, dtype=np.float32), k=1)
            logits = model(Tensor(np.array([ctx])), m).data[0, -1]
            z = logits / temp
            z = z - z.max()
            p = np.exp(z); p /= p.sum()
            out.append(int(rng.choice(len(p), p=p)))
    return "".join(rev.get(i, "?") for i in out)

for prompt in ["用户：", "用户：今天好累啊\n猫娘：", "【场景】"]:
    print("=" * 60)
    print(f"[prompt] {prompt!r}")
    print(generate(prompt, 140, temp=0.75, seed=1))
