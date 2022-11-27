from .module import Module
from ..functional import *
import numpy as np

class MSE(Module):
    def __init__(self):
        pass

    def __call__(self, *args):
        self.forward(args)

    def forward(self, *args):
        pass

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return cross_entropy(self.a, self.y)