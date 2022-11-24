import numpy as np

class Tensor:
<<<<<<< Updated upstream
    def __init__(self, *args, device=None):
        if(isinstance(args[0], list) or isinstance(args[0], np.ndarray)):
            self.item = np.array(args[0], float)
=======
    def __init__(self, *args, device=None, requires_grad=False, grad_fn=None, is_leaf=True):
        if(len(args) == 0):
            self.item = np.random.randn(1)
        elif(isinstance(args[0], list) or isinstance(args[0], np.ndarray) or isinstance(args[0], float)):
            self.item = np.array(args[0], float).round(4)
        elif(isinstance(args[0], Tensor)):
            self.item = args[0].item
>>>>>>> Stashed changes
        else:
            self.item = np.random.random(args)

        self.device = device


    def __str__(self):
        return 'tensor({})'.format(self.item)

    def __repr__(self):
<<<<<<< Updated upstream
        return 'tensor({})'.format(self.item)
=======
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

    def __rmul__(self, other):
        return self.__mul__(other)

    # 重载 @ 运算符
    def __matmul__(self, other):
        if(not isinstance(other, Tensor)):
            other = Tensor([other])
        requires_grad = self.requires_grad or other.requires_grad
        if(requires_grad):
            grad_fn = function.DotBackward(self, other, requires_grad=requires_grad)
        else:
            grad_fn = None
        
        try:
            res = self.item @ other.item
        except:
            raise ValueError('mat1 and mat2 日妈不匹配 ({} and {})'.format(self.item.shape, other.item.shape))

        return Tensor(res, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

    def __rmatmul__(self, other):
        return self.__matmul__(other)

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
>>>>>>> Stashed changes
