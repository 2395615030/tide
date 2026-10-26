import sys, os
sys.path.insert(0, r'D:\ROOT\Projects\DSH\Def\nano-torch')
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

live = np.load("D:/ROOT/Projects/DSH/Def/nano-torch/tests/traj_live.npy")
stim = np.load("D:/ROOT/Projects/DSH/Def/nano-torch/tests/traj_stim.npy")

W, H = 1000, 520
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
try:
    f = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    fs = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 11)
except Exception:
    f = ImageFont.load_default(); fs = f
COL = [(220,60,60),(60,90,220),(40,160,90),(230,150,30)]
NAMES = ["arousal","stress","novelty","satiety"]

def panel(x0, y0, w, h, data, title, vmax=2.0):
    d.rectangle([x0, y0, x0+w, y0+h], fill=(250,250,252), outline=(200,200,200))
    n = len(data)
    for i in range(4):
        pts = [(x0 + w*t/(n-1), y0+h - h*min(data[t,i],vmax)/vmax) for t in range(n)]
        d.line(pts, fill=COL[i], width=1 if n > 2000 else 2)
    d.text((x0+6, y0-18), title, font=f, fill=(30,30,30))
    for i in range(4):
        d.line([x0+w-90, y0+8+i*16, x0+w-72, y0+8+i*16], fill=COL[i], width=2)
        d.text((x0+w-66, y0+1+i*16), NAMES[i], font=fs, fill=(80,80,80))

# 上: 活系统无输入, 取 2000 tick 的一段
panel(40, 60, 440, 200, live[8000:9000], "live, no input (one stretch)")
# 下: 有刺激
panel(40, 320, 440, 160, stim, "live, stimulus at t=500 (novelty) then sustained error")
# 右: 相图 x-y
x0, y0, w, h = 540, 60, 420, 420
d.rectangle([x0, y0, x0+w, y0+h], fill=(250,250,252), outline=(200,200,200))
xs = live[8000:10000, 0]; ys = live[8000:10000, 1]
lo_x, hi_x = min(xs)-0.1, max(xs)+0.1
lo_y, hi_y = min(ys)-0.1, max(ys)+0.1
pts = [(x0 + w*(xs[t]-lo_x)/(hi_x-lo_x), y0+h - h*(ys[t]-lo_y)/(hi_y-lo_y)) for t in range(len(xs))]
d.line(pts, fill=(120,120,200), width=1)
d.text((x0+6, y0-18), "phase portrait (arousal vs stress, 2000 ticks)", font=f, fill=(30,30,30))

d.text((40, 495), "top: no input -> still moving.  bottom: stimulus perturbs phase, system keeps oscillating.", font=fs, fill=(100,100,100))
img.save("D:/ROOT/Projects/DSH/Def/nano-torch/tests/live_endo.png")
print("saved")
