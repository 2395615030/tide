"""nano-torch: 一个不依赖 PyTorch 的最小深度学习栈。

手写张量、自动微分、算子、优化器、分块存储。
上层提供两种序列模型: 参考用 attention (TinyLM) 和 多尺度门控状态 (StateLM)。
"""
from .core.tensor import (
    Tensor, add, sub, mul, div, matmul, transpose, reshape, sum_, mean,
    relu, gelu, silu, tanh, sigmoid, layernorm, softmax, cross_entropy,
    embedding, cat, numeric_grad,
)
from .core.optim import AdamW, SGD
from .core.store import save_paged, PagedMatrix, CachePool
from .core.endocrine import Endocrine
from .core.live_endo import LiveEndocrine
from .nn.statelm import StateLM, MultiScaleState, StateBlock
from .nn.gatedstate import GatedState
from .nn.transformer import TinyLM

__version__ = "0.1.0"
__all__ = [
    "Tensor", "add", "sub", "mul", "div", "matmul", "transpose", "reshape",
    "sum_", "mean", "relu", "gelu", "silu", "tanh", "sigmoid", "layernorm",
    "softmax", "cross_entropy", "embedding", "cat", "numeric_grad",
    "AdamW", "SGD", "save_paged", "PagedMatrix", "CachePool",
    "Endocrine", "LiveEndocrine",
    "StateLM", "MultiScaleState", "StateBlock", "GatedState", "TinyLM",
]