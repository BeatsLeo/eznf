import numpy as np

class Tensor:
    def __init__(self, *args, device=None, requires_grad=False, grad_fn=None, is_leaf=True):
        if(len(args) == 0):
            self.item = np.random.randn(1)
        elif(isinstance(args[0], list) or isinstance(args[0], np.ndarray) or isinstance(args[0], float)):
            self.item = np.array(args[0], float).round(4)
        elif(isinstance(args[0]), Tensor):
            self.item = args[0].item
        else:
            self.item = np.random.randn(*args)

        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.is_leaf = is_leaf
        self.device = device
        self.grad = None

    @property
    def shape(self):
        return Tensor(np.array(self.item.shape))

    @property
    def T(self):
        return Tensor(np.array(self.item.T))

    def size(self):
        return self.item.size

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

    def copy(self):
        return Tensor(self.item.copy())

    def __str__(self):
        return 'tensor(\n{}\n)'.format(self.item)

    def __repr__(self):
        return 'tensor(\n{}\n)'.format(self.item)

    def __add__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other])
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.AddBackward(self, other, requires_grad=requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item + other.item, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __mul__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other])
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.MulBackward(self, other, requires_grad=requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item * other.item, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)
        

    def backward(self, output=None):
        if(not output):
            output = Tensor([1])
        if(self.size() != output.size()):
            raise RuntimeError('grad can be implicitly created only for scalar outputs')
        if(self.grad_fn):
            self.grad_fn.backward(output)
        else:
            if(self.grad):
                self.grad += output
            else:
                self.grad = output

from eznf.autograd import function