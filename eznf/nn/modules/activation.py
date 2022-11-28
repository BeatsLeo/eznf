from .module import Module

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return relu(x)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return sigmoid(x)

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return tanh(x)

from ..functional import *