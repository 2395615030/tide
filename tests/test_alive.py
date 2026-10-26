"""死 vs 活: 耗散系统收敛到不动点, 兴奋-抑制回路产生极限环。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from nano_torch.core.endocrine import Endocrine


def run_dissipative(ticks=5000, dt=0.25):
    e = Endocrine()
    traj = np.zeros((ticks, 4))
    for t in range(ticks):
        traj[t] = e.step(0.5, 0.0, dt)   # 恒定刺激, 无 novelty
    return traj


def run_ei(ticks=20000, dt=0.01):
    """Hopf 标准形: r -> 1 的极限环. 用来证明"持续活动"是可能的。"""
    x, y = 0.05, 0.0
    omega = 1.0
    rng = np.random.RandomState(0)
    traj = np.zeros((ticks, 4))
    for t in range(ticks):
        r2 = x * x + y * y
        dx = x * (1.0 - r2) - omega * y + 0.01 * rng.randn()
        dy = y * (1.0 - r2) + omega * x + 0.01 * rng.randn()
        x += dt * dx
        y += dt * dy
        traj[t] = [x, y, -y, x]
    return traj

A = run_dissipative()
B = run_ei()

def summarize(name, traj, n=200):
    tail = traj[-n:]
    print(f"{name}: tail std per dim = " + " ".join(f"{v:.4f}" for v in tail.std(0)))
    print(f"      tail range        = " + " ".join(f"{v:.3f}" for v in (tail.max(0) - tail.min(0))))
    print(f"      drift first->last = " + " ".join(f"{v:+.4f}" for v in (tail.mean(0) - traj[:n].mean(0))))

summarize("dissipative", A)
summarize("E-I loop   ", B)

# ---- 画图 ----
W, H = 1000, 420
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
try:
    f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14); fs = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 11)
except Exception:
    f = ImageFont.load_default(); fs = f

def panel(x0, y0, w, h, data, title, colors, labels):
    d.rectangle([x0, y0, x0+w, y0+h], fill=(250,250,252), outline=(200,200,200))
    n = len(data)
    lo, hi = data.min(), data.max()
    span = max(hi - lo, 1e-6)
    for i in range(len(colors)):
        pts = [(x0 + w * t/(n-1), y0+h - h*(data[t, i]-lo)/span) for t in range(n)]
        d.line(pts, fill=colors[i], width=2)
    d.text((x0+6, y0-18), title, font=f, fill=(30,30,30))
    # 图例
    for i, lab in enumerate(labels):
        d.line([x0+w-120, y0+10+i*16, x0+w-100, y0+10+i*16], fill=colors[i], width=2)
        d.text((x0+w-94, y0+3+i*16), lab, font=fs, fill=(80,80,80))

COL = [(220,60,60),(60,90,220),(40,160,90),(230,150,30)]
panel(40, 60, 440, 300, A[:800], "A: dissipative (dead) — decays to fixed point", COL, ["arousal","stress","novelty","satiety"])
panel(520, 60, 440, 300, B[:800], "B: Hopf oscillator (alive) — limit cycle", COL, ["x","y","-y","x"])
d.text((40, 380), "A: 5000 ticks, no input -> frozen.   B: limit cycle under noise, 800 of 20000 ticks shown", font=fs, fill=(100,100,100))

# 相图
img.save("D:/ROOT/Projects/DSH/Def/nano-torch/assets/alive_dead.png")
print("saved alive_dead.png")
