import eznf
import numpy as np
from .module import Module
from ..functional import linear

class Linear(Module):
    def __init__(self, in_features, out_features, device='cpu'):
        super().__init__()
        w = 0.1*np.random.randn(out_features, in_features)
        self.w = eznf.Tensor(w, device=device, requires_grad=True, is_leaf=True)

    def forward(self, x):
        return linear(x, self.w)