import eznf
from .module import Module
from eznf.autograd import function


class Flatten(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if(x.requires_grad):
            grad_fn = function.FlattenBackward(x=x, requires_grad=x.requires_grad)
        else:
            grad_fn = None

        tensor = eznf.Tensor(grad_fn=grad_fn, requires_grad=x.requires_grad, is_leaf=False)
        tensor.item = x.item.reshape([x.shape[0], -1], order='C').T
        return tensor