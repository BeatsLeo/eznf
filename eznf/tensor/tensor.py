import numpy as np
import cupy as cpy

class Tensor:
    def __init__(self, *args, device=None, requires_grad=False, grad_fn=None, is_leaf=True):
        arr = cpy if(device == 'gpu') else np

        if(len(args) == 0):
            self.item = arr.random.randn(1)
        elif(isinstance(args[0], list) or isinstance(args[0], float)):
            self.item = arr.array(args[0], float)
        elif(isinstance(args[0], arr.ndarray)):
            if(args[0].dtype == bool):
                self.item = arr.array(args[0])
            else:
                self.item = args[0]
        elif(isinstance(args[0], Tensor)):
            self.item = args[0].item
        else:
            self.item = arr.random.randn(*args)

        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.is_leaf = is_leaf
        self.device = device if(device == 'gpu') else 'cpu'
        self.arr = arr
        self.grad = None

    # GPU或CPU之间转换
    def to(self, device: str):
        if(device.lower() == 'gpu' and self.device.lower() != 'gpu'):
            self.item = cpy.array(self.item)
            self.device = 'gpu'
        elif(self.device.lower() == 'gpu'):
            self.item = self.item.get()
            self.device = 'cpu'
        
        return self

    @property
    def shape(self):
        return self.item.shape

    @property
    def T(self):
        if(self.requires_grad):
            grad_fn = function.PermuteBackward(self, requires_grad=self.requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item.T, device=self.device, grad_fn=grad_fn, requires_grad=self.requires_grad, is_leaf=self.is_leaf)

    def size(self):
        return self.item.size

    def dim(self):
        return len(self.item.shape)

    def max(self, axis=None):
        tensor = Tensor(device=self.device, is_leaf=False)
        tensor.item = self.item.max(axis=axis)
        return tensor

    def min(self, axis=None):
        tensor = Tensor(device=self.device, is_leaf=False)
        tensor.item = self.item.min(axis=axis)
        return tensor

    def sum(self, axis=None):
        if(self.requires_grad):
            grad_fn = function.SumBackward(self, requires_grad=self.requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item.sum(axis=axis), device=self.device, requires_grad=self.requires_grad, grad_fn=grad_fn, is_leaf=False)

    def mean(self, axis=None):
        return Tensor(self.item.mean(axis=axis), device=self.device, is_leaf=False)

    def var(self, axis=None):
        if(axis != None):
            n = self.item.shape[axis]
        else:
            n = self.item.size
        return Tensor(self.arr.array(self.item.var(axis=axis)*n/(n-1)), device=self.device, is_leaf=False)

    def std(self, axis=None):
        if(axis != None):
            n = self.item.shape[axis]
        else:
            n = self.item.size
        return self.var().sqrt()

    def abs(self):
        return Tensor(self.arr.abs(self.item), device=self.device, is_leaf=False)

    def argmin(self, axis=None):
        tensor = Tensor(device=self.device, is_leaf=False)
        tensor.item = self.item.argmin(axis=axis)
        return tensor
    
    def argmax(self, axis=None):
        tensor = Tensor(device=self.device, is_leaf=False)
        tensor.item = self.item.argmax(axis=axis)
        return tensor

    def view(self, *args):
        if(self.requires_grad):
            grad_fn = function.PermuteBackward(self, requires_grad=self.requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item.reshape(*args), device=self.device, grad_fn=grad_fn, requires_grad=self.requires_grad, is_leaf=self.is_leaf)

    def sqrt(self):
        return Tensor(self.arr.sqrt(self.item), device=self.device, is_leaf=False)

    def tolist(self):
        return self.item.tolist()

    def numpy(self):
        if(self.device == 'gpu'):
            return self.item.get()
        return self.item

    def sin(self):
        return Tensor(self.arr.sin(self.item), device=self.device, is_leaf=False)

    def cos(self):
        return Tensor(self.arr.cos(self.item), device=self.device, is_leaf=False)

    def tan(self):
        return Tensor(self.arr.tan(self.item), device=self.device, is_leaf=False)

    def exp(self):
        if(self.requires_grad):
            grad_fn = function.ExpBackward(self, requires_grad=self.requires_grad)
        else:
            grad_fn = None
        return Tensor(self.arr.exp(self.item), device=self.device, requires_grad=self.requires_grad, grad_fn=grad_fn, is_leaf=False)

    def copy(self):
        return Tensor(self.item.copy(), device=self.device, requires_grad=self.requires_grad, grad_fn=self.grad_fn, is_leaf=self.is_leaf)

    def __str__(self):
        return 'tensor(\n{}\n)'.format(self.item)

    def __repr__(self):
        return 'tensor(\n{}\n)'.format(self.item)
    
    def __len__(self):
        return len(self.item)

    def __getitem__(self, *args):
        return Tensor(self.arr.array(self.item.__getitem__(*args)), device=self.device, requires_grad=self.requires_grad, is_leaf=False)

    def __setitem__(self, key, value):
        self.item.__setitem__(key, value)

    def __lt__(self, other):
        return Tensor(self.arr.array(self.item.__lt__(other)), device=self.device, requires_grad=self.requires_grad, is_leaf=False)
    
    def __le__(self, other):
        return Tensor(self.arr.array(self.item.__le__(other)), device=self.device, requires_grad=self.requires_grad, is_leaf=False)

    def __eq__(self, other):
        return Tensor(self.arr.array(self.item.__eq__(other)), device=self.device, requires_grad=self.requires_grad, is_leaf=False)

    def __ne__(self, other):
        return Tensor(self.arr.array(self.item.__ne__(other)), device=self.device, requires_grad=self.requires_grad, is_leaf=False)

    def __gt__(self, other):
        return Tensor(self.arr.array(self.item.__gt__(other)), device=self.device, requires_grad=self.requires_grad, is_leaf=False)
        
    def __ge__(self, other):
        return Tensor(self.arr.array(self.item.__ge__(other)), device=self.device, requires_grad=self.requires_grad, is_leaf=False)

    def __neg__(self):
        if(self.requires_grad):
            grad_fn = function.NegBackward(self, requires_grad=self.requires_grad)
        else:
            grad_fn = None
        return Tensor(-self.item, device=self.device, requires_grad=self.requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __add__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other], device=self.device)
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.AddBackward(self, other, requires_grad=requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item + other.item, device=self.device, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other], device=self.device)
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.SubBackward(self, other, requires_grad=requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item - other.item, device=self.device, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other], device=self.device)
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.MulBackward(self, other, requires_grad=requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item * other.item, device=self.device, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other], device=self.device)
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.DivBackward(self, other, requires_grad=requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item / other.item, device=self.device, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __rtruediv__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other], device=self.device)
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.DivBackward(other, self, requires_grad=requires_grad)
        else:
            grad_fn = None
        return Tensor(other.item / self.item, device=self.device, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    # 重载 @ 运算符
    def __matmul__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other], device=self.device)
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.DotBackward(self, other, requires_grad=requires_grad)
        else:
            grad_fn = None
        # 检测维度
        try:
            res = self.item @ other.item
        except:
            raise ValueError('mat1 and mat2 shapes cannot be multiplied ({} and {})'.format(self.item.shape, other.item.shape))

        return Tensor(res, requires_grad=requires_grad, device=self.device, grad_fn=grad_fn, is_leaf=False)

    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def __pow__(self, other: int):
        if(self.requires_grad):
            grad_fn = function.PowBackward(self, other, requires_grad=self.requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item**other, device=self.device, requires_grad=self.requires_grad, grad_fn=grad_fn, is_leaf=False)

    def backward(self, output=None):
        if(not output):
            output = Tensor([1], device=self.device, is_leaf=False)
        if(self.size() != output.size()):
            raise RuntimeError('grad can be implicitly created only for scalar outputs')
        if(self.grad_fn):
            self.grad_fn.backward(output)
            self.grad_fn = None
        elif(self.is_leaf):
            if(self.grad):
                self.grad += output
            else:
                self.grad = output


from eznf.autograd import function