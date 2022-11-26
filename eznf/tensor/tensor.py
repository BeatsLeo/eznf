import numpy as np

class Tensor:
    def __init__(self, *args, device=None, requires_grad=False, grad_fn=None, is_leaf=True):
        if(len(args) == 0):
            self.item = np.random.randn(1)
        elif(isinstance(args[0], list) or isinstance(args[0], float)):
            self.item = np.array(args[0], float).round(4)
        elif(isinstance(args[0], np.ndarray)):
            self.item = np.array(args[0])
        elif(isinstance(args[0], Tensor)):
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
        return self.item.shape

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
        tensor = Tensor(is_leaf=False)
        tensor.item = self.item.argmin(axis=axis)
        return tensor
    
    def argmax(self, axis=None):
        tensor = Tensor(is_leaf=False)
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
        return Tensor(np.sin(self.item), is_leaf=False)

    def cos(self):
        return Tensor(np.cos(self.item), is_leaf=False)

    def tan(self):
        return Tensor(np.tan(self.item), is_leaf=False)

    def tanh(self):
        return Tensor(np.tanh(self.item), is_leaf=False)

    def exp(self):
        return Tensor(np.exp(self.item), is_leaf=False)

    def copy(self):
        return Tensor(self.item.copy())

    def __str__(self):
        return 'tensor(\n{}\n)'.format(self.item)

    def __repr__(self):
<<<<<<< HEAD
        return 'tensor(\n{}\n)'.format(self.item)
    
    def __len__(self):
        return len(self.item)

    def __getitem__(self, *args):
        return Tensor(self.item.__getitem__(*args), is_leaf=False)

    def __setitem__(self, key, value):
        self.item.__setitem__(key, value)

    def __lt__(self, other):
        return Tensor(self.item.__lt__(other), is_leaf=False)
    
    def __le__(self, other):
        return Tensor(self.item.__le__(other), is_leaf=False)

    def __eq__(self, other):
        return Tensor(self.item.__eq__(other), is_leaf=False)

    def __ne__(self, other):
        return Tensor(self.item.__ne__(other), is_leaf=False)

    def __gt__(self, other):
        return Tensor(self.item.__gt__(other), is_leaf=False)
        
    def __ge__(self, other):
        return Tensor(self.item.__ge__(other), is_leaf=False)

    def __neg__(self):
        if(self.requires_grad):
            grad_fn = function.NegBackward(self, requires_grad=self.requires_grad)
        else:
            grad_fn = None
        return Tensor(-self.item, requires_grad=self.requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __add__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other])
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.AddBackward(self, other, requires_grad=requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item + other.item, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other])
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.SubBackward(self, other, requires_grad=requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item - other.item, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other])
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.MulBackward(self, other, requires_grad=requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item * other.item, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other])
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.DivBackward(self, other, requires_grad=requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item / other.item, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __rtruediv__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other])
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.DivBackward(other, self, requires_grad=requires_grad)
        else:
            grad_fn = None
        return Tensor(other.item / self.item, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    # 重载 @ 运算符
    def __matmul__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other])
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

        return Tensor(res, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __rmatmul__(self, other):
        return self.__matmul__(other)

    def __pow__(self, other: int):
        if(self.requires_grad):
            grad_fn = function.PowBackward(self, other, requires_grad=self.requires_grad)
        else:
            grad_fn = None
        return Tensor(self.item**other, requires_grad=self.requires_grad, grad_fn=grad_fn, is_leaf=False)

    def backward(self, output=None):
        if(not output):
            output = Tensor([1], is_leaf=False)
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
=======
        return 'tensor({})'.format(self.item)
>>>>>>> fae521059b7860c9cbd901d5e0107c95b96afeaa
