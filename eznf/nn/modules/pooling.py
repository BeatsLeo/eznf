import eznf
import numpy as np
from .module import Module
from ..functional import max_pooling

class MaxPooling(Module):
    # 仅支持stride = 1, padding = 0
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return max_pooling(x, self.kernel_size)