import sys, os
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from nano_torch.core.tensor import Tensor
from nano_torch.nn.statelm import StateLM
from nano_torch.data.neko import load_neko

text, ids, keep = load_neko()
V = len(keep) + 1
for T in (96, 384):
    np.random.seed(42)
    model = StateLM(V, d=128, n_layers=3, max_T=T, gated=True)
    x = np.random.randint(0, V, size=(8, T))
    _ = model(Tensor(x))
    for li, b in enumerate(model.blocks):
        mx = b.mix
        g = mx.last_g
        pre = mx.last_pre
        print(f"T={T} layer{li}: g mean={g.mean():.4f} std={g.std():.4f} "
              f"min={g.min():.4f} max={g.max():.4f}  |pre| mean={np.abs(pre).mean():.4f} "
              f"max={np.abs(pre).max():.4f}")
    print()
