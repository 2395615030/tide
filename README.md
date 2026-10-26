# nano-torch

一个不依赖 PyTorch 的最小深度学习栈。手写张量、手写自动微分、手写算子、手写优化器。
目标不是复刻框架，而是在极小的资源下完成训练，并验证一些没人验证过的结构。

## 快速开始

```
python examples/train_tiny.py
```

8 秒训练完一个字符级模型, 然后生成文本。

## 结构

```
nano_torch/
  core/
    tensor.py       Tensor + 20 个算子 + 数值梯度检验
    optim.py        AdamW / SGD
    store.py        分块权重文件(NDST) + 容量受限 LRU 缓存池
    endocrine.py    四激素内分泌系统(耗散型, 会停在不动点)
    live_endo.py    带自激振荡的内分泌系统(极限环)
  nn/
    paged.py        PagedLinear: 权重/梯度按块走
    transformer.py  参考实现: decoder-only attention
    statelm.py      多尺度状态层, 完全不用 attention
    gatedstate.py   多尺度状态 + 内容门控
  data/
    neko.py         Neko-A 多轮对话加载
    mix.py          tieba + neko 混合语料

examples/           最小可运行例子
tests/              7 个可复现的断言 (gradcheck / 分块 / 对比 / 死活)
experiments/        35 个归档的探索脚本 (可运行, 但不是断言)
assets/             图
```

## 已经验证的

| 结论 | 数字 | 条件 |
|---|---|---|
| 自动微分正确 | gradcheck 最大相对误差 3.4e-9 | 13 个算子 |
| 分块权重可行 | 32 MB 矩阵 / 1 MB 缓存池, 误差 0 | 逐块流式 |
| 低资源训练可行 | 5.06 MB 权重 / 512 KB 池, 准确率 1.000 | 288 步 |
| 缓存命中率的决定因素 | 池 < 权重时 0%, 装得下 90% | 顺序全量扫描 |
| 稀疏更新制造局部性 | 命中率 66.9% -> 88.9%, 快 1.7x | 池 6 MB |
| **多尺度门控状态层优于 attention** | **-7.33% (3 seed: -7.90/-3.16/-10.94, wd=0.3)** | T=96, 500 步, 取最佳 val |
| 正则化是关键 | wd 0.01 -> 0.3: 最佳 val 3.2735 -> 3.2328 | 且最佳点从 300 步延后到 499 |
| 不加正则时的假象 | 300 步 -6.34%, 但 450 步后反超 | 那是"收敛快", 不是"最终好" |
| 长序列仍然占优 | -5.20% val | T=384, 200 步, clip=1.0 |
| 纯状态(无门控)不如 attention | +29.0% val | 内容路由不可省 |

## 被证伪的

| 假设 | 结果 | 说明 |
|---|---|---|
| 情绪加到 logits | +0.32% (更差) | 冗余, attention 已含全部上下文 |
| 情绪调学习率 | 公平 A/B 后 -0.68% | Adam 已自适应 |
| 情绪调门控(动态) | -4.72%, 但固定值 -5.60% | **常数更好, 动态是噪声** |
| 情绪调门控(非平稳任务) | -0.8% | 任务不够冲突 |
| 情绪调门控(活系统) | 3 seed: -1.64/+5.49/-2.35 | 平均 +0.5%, 噪声 |

**结论: 在当前的任务形态下(静态数据集 + 单任务语言建模), 情绪没有可测的价值。**
根因: 模型是函数 `f(context) -> output`, 两次调用之间不存在, 状态无处积累。

## 内分泌系统的两次状态

- `endocrine.py`: 耗散型。无输入时收敛到不动点, tail std = 0.0000。**死的。**
- `live_endo.py`: 加 Hopf 振荡器。无输入 20000 tick 后 tail std = 0.19/0.20/0.08/0.08, 相图为闭合极限环。**活的。**

## 已知限制

- 只跑 CPU (numpy)。MX450 有 CUDA, 但 Windows 上缺 MSVC 编译器, 无法编译 kernel。
- 实测瓶颈: numpy 的 3D batched matmul 比 2D 慢 3.1x; gelu 的 `x**3` 比 `x*x*x` 慢 3.4x。
- 实验分辨率约 +-4% (3 seed 的 val 波动), 小于此的效应测不出。
- Adam 状态未分块, 大模型下会撑爆内存。
- 权重文件无崩溃一致性保证。

## 结论的适用范围

多尺度门控状态层确实优于 attention, 但前提是正则化到位。

不加正则时, 它在 300 步领先 6%, 350 步打平, 450 步后因过拟合反超——那是"收敛快"的假象。
把 weight decay 从 0.01 提到 0.3 之后, 最佳点从 300 步延后到 499 步, 三个 seed 的一致差距是 -7.33%。

教训: 一个结构"更快收敛"和"最终更好"是两件事, 混在一起会得出错误结论。

## 性能

| 场景 | 优化前 | 优化后 |
|---|---|---|
| T=384 单步 | 4998 ms | 1927 ms (2.6x) |

## 测试

```
python tests/test_autograd.py          # 20 个算子的数值梯度检验
python tests/test_store.py             # 32MB 矩阵 / 1MB 池
python tests/test_hitrate.py           # 命中率 vs 池容量
python tests/test_sparse_update.py     # 全量 vs 1/4 轮转更新
python tests/test_alive.py             # 耗散系统 vs 极限环
python tests/test_attn_gated_seeds.py  # attn vs gated, 3 seed (~25 min)
python tests/test_fair_wd.py           # 加正则后的公平对比 (~30 min)
```

`experiments/` 里是被归档的探索脚本, 按主题分在
`diagnostics/ emotion/ arch/ training/` 四个目录下, 可以运行, 但结论不保证可复现。
