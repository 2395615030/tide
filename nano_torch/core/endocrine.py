"""人工内分泌系统: 四个腺体, 连续动力学, 先验证行为。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

NAMES = ["arousal", "stress", "novelty", "satiety"]
COLORS = [(220, 60, 60), (60, 90, 220), (40, 160, 90), (230, 150, 30)]


class Endocrine:
    """四个腺体, 各自有分泌、降解、耦合。浓度是标量, 但动力学是路径依赖的。"""
    def __init__(self):
        self.c0 = np.array([0.30, 0.10, 0.00, 0.50])
        self.c = self.c0.copy()
        self.lam = np.array([0.50, 0.03, 0.50, 0.02])   # 时间常数: 压力 ~33 步, 新奇 ~2 步

    def step(self, error, novelty, dt=0.25):
        sec = np.array([
            0.20 * max(0.0, error - 1.0) + 0.50 * max(0.0, novelty),  # arousal
            0.014 * max(0.0, error - 1.0),                             # stress
            1.20 * max(0.0, novelty),                                  # novelty
            0.030 * max(0.0, 1.0 - error),                             # satiety
        ])
        coup = np.array([
            -0.15 * self.c[1] + 0.30 * self.c[2],  # 压力压唤醒, 新奇抬唤醒
            0.0, 0.0, 0.0,
        ])
        dc = sec - self.lam * (self.c - self.c0) + coup
        self.c = np.clip(self.c + dt * dc, 0.0, 2.0)
        return self.c.copy()


def stimulus(t):
    """平静 -> 连续打击 -> 恢复"""
    if t < 40:
        return 0.5, 0.0
    if t < 50:
        return 0.5 + (t - 40) * 0.25, 0.0        # 误差快速爬升
    if t < 54:
        return 3.0, 1.0                          # 连续几步的分布外输入
    if t < 110:
        return 3.0, 0.0                          # 持续高压
    if t == 110:
        return 0.5, 0.0                          # 恢复
    return 0.5, 0.0



if __name__ == "__main__":
    endo = Endocrine()
    T = 220
    hist = np.zeros((T, 4))
    errs = np.zeros(T)
    for t in range(T):
        e, n = stimulus(t)
        errs[t] = e
        hist[t] = endo.step(e, n)

    # ---- 画图 ----
    W, H = 1000, 460
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
        fs = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
    except Exception:
        f = ImageFont.load_default(); fs = f

    L, R, TP, BT = 70, W - 30, 40, H - 60

    def X(t): return L + (R - L) * t / (T - 1)
    def Y(v): return BT - (BT - TP) * min(v, 2.5) / 2.5

    # 背景分区
    d.rectangle([L, TP, R, BT], fill=(250, 250, 252), outline=(200, 200, 200))
    for g in [0.5, 1.0, 1.5, 2.0]:
        y = Y(g)
        d.line([L, y, R, y], fill=(230, 230, 235))
        d.text((L - 34, y - 7), f"{g:.1f}", font=fs, fill=(120, 120, 120))
    for t in [40, 50, 110]:
        x = X(t)
        d.line([x, TP, x, BT], fill=(210, 210, 215), width=1)

    # 曲线
    for i in range(4):
        pts = [(X(t), Y(hist[t, i])) for t in range(T)]
        d.line(pts, fill=COLORS[i], width=3)

    # 刺激(灰色虚线)
    pts = [(X(t), Y(errs[t] * 0.6 + 0.1)) for t in range(T)]
    d.line(pts, fill=(180, 180, 180), width=2)

    # 图例
    ly = TP + 6
    for i, nm in enumerate(NAMES):
        d.line([R - 150, ly + 8, R - 120, ly + 8], fill=COLORS[i], width=3)
        d.text((R - 112, ly), nm, font=f, fill=(60, 60, 60))
        ly += 22
    d.line([R - 150, ly + 8, R - 120, ly + 8], fill=(180, 180, 180), width=2)
    d.text((R - 112, ly), "stimulus", font=f, fill=(120, 120, 120))

    d.text((L, 12), "Artificial endocrine system — concentration over time", font=f, fill=(30, 30, 30))
    d.text((X(10), BT + 12), "calm", font=fs, fill=(120, 120, 120))
    d.text((X(60), BT + 12), "sustained stress", font=fs, fill=(120, 120, 120))
    d.text((X(150), BT + 12), "recovery", font=fs, fill=(120, 120, 120))

    out = "D:/ROOT/Projects/DSH/Def/nano-torch/tests/endocrine.png"
    img.save(out)
    print("saved", out)
    print("\nt  err  arousal stress novelty satiety")
    for t in [0, 45, 50, 60, 100, 110, 120, 140, 170, 210]:
        a, s, n, sa = hist[t]
        print(f"{t:4d} {errs[t]:4.1f}  {a:.2f}   {s:.2f}   {n:.2f}   {sa:.2f}")
