from eznf.tensor.tensor import Tensor
import numpy as np

class Functional():
    def __init__(self, requires_grad=False):
        self.requires_grad = requires_grad
    
    def backward(self, *grad_outputs):
        raise NotImplementedError("You must implement either the backward method for your custom autograd.Function to use it with backward mode.")


class AddBackward(Functional):
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
    pass

class MulBackward(Functional):
    def __init__(self, a: Tensor, b: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
        self.b = b
    
    def backward(self, output = Tensor([1])):
        if(self.a.requires_grad):
            self.a.backward(Tensor([1]) * self.b * output)
        if(self.b.requires_grad):
            self.b.backward(Tensor([1]) * self.a * output)

class DivBackward(Functional):
    pass

class DotBackward(Functional):
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
