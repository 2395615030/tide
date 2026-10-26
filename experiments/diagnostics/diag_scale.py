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
    x = np.random.randint(0, V, size=(4, T))
    _ = model(Tensor(x))
    for li, b in enumerate(model.blocks):
        H = b.mix.last_H
        # 每个尺度通道的幅度
        per = [H[:, :, s*128:(s+1)*128].std() for s in range(4)]
        print(f"T={T} layer{li}: H std total={H.std():.4f}  per-scale=" +
              " ".join(f"{v:.4f}" for v in per))
    print()
