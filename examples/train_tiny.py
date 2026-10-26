"""最小例子: 用多尺度门控状态模型学一个重复文本模式。

在项目根目录运行:
    python examples/train_tiny.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import nano_torch as nt


def main():
    text = "用户：你好\n助手：你好呀，有什么可以帮你？\n" * 300
    chars = sorted(set(text))
    vocab = {c: i for i, c in enumerate(chars)}
    data = np.array([vocab[c] for c in text], dtype=np.int64)
    T, B, STEPS = 48, 16, 200

    model = nt.StateLM(len(chars), d=64, n_layers=2, max_T=T, gated=True)
    opt = nt.AdamW(model.params(), lr=3e-3, weight_decay=0.1)
    rng = np.random.RandomState(0)
    print(f"vocab {len(chars)}  params {sum(p.size for p in model.params())}")

    t0 = time.time()
    for s in range(STEPS):
        ix = rng.randint(0, len(data) - T - 1, size=B)
        x = np.stack([data[i:i+T] for i in ix])
        y = np.stack([data[i+1:i+T+1] for i in ix])
        loss = nt.cross_entropy(model(nt.Tensor(x)), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 40 == 0 or s == STEPS - 1:
            print(f"step {s:3d}  loss {float(loss.data):.4f}")

    print(f"done in {time.time()-t0:.1f}s")

    # 生成
    out = [vocab[c] for c in "用户："]
    for _ in range(40):
        ctx = np.array(out[-T:])
        logits = model(nt.Tensor(ctx[None, :])).data[0, -1]
        z = logits / 0.8
        z -= z.max()
        p = np.exp(z); p /= p.sum()
        out.append(int(rng.choice(len(p), p=p)))
    rev = {i: c for c, i in vocab.items()}
    print("sample:", "".join(rev.get(i, "?") for i in out))


if __name__ == "__main__":
    main()