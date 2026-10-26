"""活着的内分泌: 在分泌-降解之上, 加一个自激振荡器。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np


class LiveEndocrine:
    """四个激素浓度 + 一对兴奋-抑制振荡变量。

    振荡器: Hopf 标准形, r -> 1 的极限环。
    激素: c' = sec - lam*(c-c0) + k_osc * oscillator
    """
    def __init__(self, seed=0, omega=0.7):
        self.c0 = np.array([0.30, 0.10, 0.00, 0.50])
        self.c = self.c0.copy()
        self.lam = np.array([0.50, 0.03, 0.50, 0.02])
        self.k_osc = np.array([0.30, 0.20, 0.15, 0.08])
        self.omega = omega
        self.x, self.y = 0.6, 0.0
        self.rng = np.random.RandomState(seed)

    def step(self, error=0.0, novelty=0.0, dt=0.05):
        # --- 振荡器 (带噪声, 避免完全周期) ---
        r2 = self.x * self.x + self.y * self.y
        noise = 0.02 * self.rng.randn()
        stim = 0.30 * novelty + 0.05 * max(0.0, error - 1.0)
        dx = self.x * (1.0 - r2) - self.omega * self.y + noise
        dy = self.y * (1.0 - r2) + self.omega * self.x + noise + stim
        self.x += dt * dx
        self.y += dt * dy

        # --- 激素 ---
        sec = np.array([
            0.20 * max(0.0, error - 1.0) + 0.50 * max(0.0, novelty),
            0.014 * max(0.0, error - 1.0),
            1.20 * max(0.0, novelty),
            0.030 * max(0.0, 1.0 - error),
        ])
        coup = np.array([-0.15 * self.c[1] + 0.30 * self.c[2], 0.0, 0.0, 0.0])
        osc = self.k_osc * np.array([self.x, -self.y, self.y, -self.x])
        dc = sec - self.lam * (self.c - self.c0) + coup + osc
        self.c = np.clip(self.c + dt * dc, 0.0, 2.0)
        return self.c.copy()


if __name__ == "__main__":
    e = LiveEndocrine()
    T = 20000
    traj = np.zeros((T, 4))
    for t in range(T):
        traj[t] = e.step()          # 无输入
    tail = traj[-2000:]
    print("tail std  :", " ".join(f"{v:.4f}" for v in tail.std(0)))
    print("tail range:", " ".join(f"{v:.4f}" for v in (tail.max(0) - tail.min(0))))
    print("tail mean :", " ".join(f"{v:.3f}" for v in tail.mean(0)))

    # 有刺激时
    e2 = LiveEndocrine(seed=1)
    traj2 = np.zeros((2000, 4))
    for t in range(2000):
        err = 3.0 if 500 <= t < 1500 else 0.5
        nov = 1.0 if t == 500 else 0.0
        traj2[t] = e2.step(err, nov)
    print("\nwith stimulus: std", " ".join(f"{v:.4f}" for v in traj2[500:1500].std(0)))
    np.save("D:/ROOT/Projects/DSH/Def/nano-torch/tests/traj_live.npy", traj)
    np.save("D:/ROOT/Projects/DSH/Def/nano-torch/tests/traj_stim.npy", traj2)
    print("saved trajectories")
