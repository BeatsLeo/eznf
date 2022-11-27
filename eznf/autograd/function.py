from eznf.tensor.tensor import Tensor
import numpy as np

class Functional():
    def __init__(self, requires_grad=False):
        self.requires_grad = requires_grad
    
    def backward(self, *grad_outputs):
        raise NotImplementedError("You must implement either the backward method for your custom autograd.Function to use it with backward mode.")

class NegBackward(Functional):
    # -a
    def __init__(self, a: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
    
    def backward(self, output = Tensor([1])):
        if(self.a.requires_grad):
            self.a.backward(-Tensor(np.ones_like(self.a.item), is_leaf=False) * output)

class PermuteBackward(Functional):
    # a.T
    def __init__(self, a: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
    
    def backward(self, output = Tensor([1])):
        if(self.a.requires_grad):
            self.a.backward(Tensor(np.ones_like(self.a.item), is_leaf=False) * output.T)

class SumBackward(Functional):
    # sum(a)
    def __init__(self, a: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
    
    def backward(self, output = Tensor([1])):
        if(self.a.requires_grad):
            self.a.backward(Tensor(np.ones_like(self.a.item), is_leaf=False) * output)

class AddBackward(Functional):
    # a + b
    def __init__(self, a: Tensor, b: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
        self.b = b
    
    def backward(self, output = Tensor([1])):
        if(self.a.requires_grad):
            self.a.backward(Tensor(np.ones_like(self.a.item), is_leaf=False) * output)
        if(self.b.requires_grad):
            self.b.backward(Tensor(np.ones_like(self.b.item), is_leaf=False) * output)

class SubBackward(Functional):
    # a - b
    def __init__(self, a: Tensor, b: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
        self.b = b
    
    def backward(self, output = Tensor([1])):
        if(self.a.requires_grad):
            self.a.backward(Tensor(np.ones_like(self.a.item), is_leaf=False) * output)
        if(self.b.requires_grad):
            self.b.backward(-Tensor(np.ones_like(self.b.item), is_leaf=False) * output)

class MulBackward(Functional):
    # a * b
    def __init__(self, a: Tensor, b: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
        self.b = b
    
    def backward(self, output = Tensor([1])):
        if(self.a.requires_grad):
            self.a.backward(self.b * output)
        if(self.b.requires_grad):
            self.b.backward(self.a * output)

class DivBackward(Functional):
    # a / b
    def __init__(self, a: Tensor, b: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
        self.b = b
    
    def backward(self, output = Tensor([1])):
        if(self.a.requires_grad):
            self.a.backward(1 / self.b * output)
        if(self.b.requires_grad):
            self.b.backward(self.a * (-1/self.b**2) * output)

class DotBackward(Functional):
    # a @ b
    def __init__(self, a: Tensor, b: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
        self.b = b
    
    def backward(self, output = Tensor([1])):
        if(self.a.requires_grad):
            if(output.size() == 1):
                self.a.backward(output * self.b.T)
            else:
                self.a.backward(output @ self.b.T)

        if(self.b.requires_grad):
            if(output.size() == 1):
                self.b.backward(self.a.T * output)
            else:
                self.b.backward(self.a.T @ output)

class PowBackward(Functional):
    # a**b
    def __init__(self, a: Tensor, b: int, requires_grad=False):
        super().__init__()
        self.a = a
        self.b = b
    
    def backward(self, output = Tensor([1])):
        if(self.a.requires_grad):
            self.a.backward(self.b * self.a**(self.b-1) * output)

class ReluBackward(Functional):
    def __init__(self, x: Tensor, requires_grad=False):
        super().__init__()
        self.x = x
    
    def backward(self, output = Tensor([1])):
       if(self.x.requires_grad):
            self.x.backward((self.x > 0) * output)

class SoftmaxBackward(Functional):
    def __init__(self, x: Tensor, axis: int, requires_grad=False):
        super().__init__()
        self.x = x
        self.axis = axis
    
    def backward(self, output = Tensor([1])):
        def softmax(x, axis):
            phi = x.max(axis=axis)
            e = np.exp(x - phi)
            return e / e.sum(axis=axis)

        s = softmax(self.x.item, self.axis)
        res = Tensor(s*(s-1), requires_grad=False, is_leaf=False)
        if(self.x.requires_grad):
            self.x.backward(res * output)

class CrossEntropyBackward(Functional):
    def __init__(self, x: Tensor, y: Tensor, requires_grad=False):
        super().__init__()
        self.x = x
        self.y = y
    
    def backward(self, output = Tensor([1])):
        def softmax(x):
            phi = x.max(axis=0)
            e = np.exp(x - phi)
            return e / e.sum(axis=0)
        a = softmax(self.x.item)
        res = Tensor(a - self.y.item, requires_grad=False, is_leaf=False)
        if(self.x.requires_grad):
            self.x.backward(res * output)