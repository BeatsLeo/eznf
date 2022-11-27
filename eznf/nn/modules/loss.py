from .module import Module
from ..functional import *

class MSE(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return mse(x, y)

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return cross_entropy(x, y)