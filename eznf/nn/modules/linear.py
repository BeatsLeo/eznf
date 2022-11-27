from .module import Module
import eznf

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        w = 0.1*eznf.randn(out_features, in_features)
        self.w = eznf.Tensor(w, requires_grad=True, is_leaf=True)

    def forward(self, x):
        return self.w @ x