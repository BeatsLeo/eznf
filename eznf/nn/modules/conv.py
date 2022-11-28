import eznf
import numpy as np
from .module import Module
from ..functional import cov2d

class Cov2d(Module):
    # 仅支持stride = 1, padding = 0
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        w = 0.1*np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.w = eznf.Tensor(w, requires_grad=True, is_leaf=True)

    def forward(self, x):
        return cov2d(x, self.w)