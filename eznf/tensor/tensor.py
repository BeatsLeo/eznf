import numpy as np
class Tensor:
    def __init__(self, *args, device=None, requires_grad=False):
        if(len(args) == 0):
            self.item = np.random.randn(1)
        elif(isinstance(args[0], list) or isinstance(args[0], np.ndarray) or isinstance(args[0], float)):
            self.item = np.array(args[0], float).round(4)
        else:
            self.item = np.random.randn(*args)

        self.requires_grad = requires_grad
        self.grad_fn = None
        self.device = device

    @property
    def shape(self):
        return Tensor(np.array(self.item.shape))

    @property
    def T(self):
        return Tensor(np.array(self.item.T))

    def size(self):
        return Tensor(np.array(self.item.shape))

    def dim(self):
        return len(self.item.shape)

    def mean(self, axis=None):
        return Tensor(self.item.mean(axis=axis).round(4))

    def var(self, axis=None):
        if(axis != None):
            n = self.item.shape[axis]
        else:
            n = self.item.size
        return Tensor(np.array(self.item.var(axis=axis)*n/(n-1)).round(4))

    def std(self, axis=None):
        if(axis != None):
            n = self.item.shape[axis]
        else:
            n = self.item.size
        return self.var().sqrt()

    def abs(self):
        return Tensor(np.abs(self.item))

    def argmin(self, axis=None):
        tensor = Tensor()
        tensor.item = self.item.argmin(axis=axis)
        return tensor
    
    def argmax(self, axis=None):
        tensor = Tensor()
        tensor.item = self.item.argmax(axis=axis)
        return tensor

    def view(self, *args):
        return Tensor(self.item.reshape(*args))

    def sqrt(self):
        return Tensor(np.sqrt(self.item))

    def tolist(self):
        return self.item.tolist()

    def numpy(self):
        return self.item

    def sin(self):
        return Tensor(np.sin(self.item))

    def cos(self):
        return Tensor(np.cos(self.item))

    def tan(self):
        return Tensor(np.tan(self.item))

    def tanh(self):
        return Tensor(np.tanh(self.item))

    def exp(self):
        return Tensor(np.exp(self.item))

    def mm(self, mat2):
        if(not isinstance(mat2, Tensor)):
            raise TypeError("mat2 must be a Tensor")
        return Tensor(np.matmul(self.item, mat2))

    def __str__(self):
        return 'tensor(\n{}\n)'.format(self.item)

    def __repr__(self):
        return 'tensor(\n{}\n)'.format(self.item)