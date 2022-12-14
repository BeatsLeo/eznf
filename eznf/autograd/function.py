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
        output.to(self.a.device)
        if(self.a.requires_grad):
            self.a.backward(-Tensor(self.a.arr.ones_like(self.a.item), device=self.a.device, is_leaf=False) * output)

class PermuteBackward(Functional):
    # a.T
    def __init__(self, a: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
    
    def backward(self, output = Tensor([1])):
        output.to(self.a.device)
        if(self.a.requires_grad):
            self.a.backward(Tensor(self.a.arr.ones_like(self.a.item), device=self.a.device, is_leaf=False) * output.T)

class SumBackward(Functional):
    # sum(a)
    def __init__(self, a: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
    
    def backward(self, output = Tensor([1])):
        output.to(self.a.device)
        if(self.a.requires_grad):
            self.a.backward(Tensor(self.a.arr.ones_like(self.a.item), device=self.a.device, is_leaf=False) * output)

class AddBackward(Functional):
    # a + b
    def __init__(self, a: Tensor, b: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
        self.b = b
    
    def backward(self, output = Tensor([1])):
        output.to(self.a.device)
        if(self.a.requires_grad):
            self.a.backward(Tensor(self.a.arr.ones_like(self.a.item), device=self.a.device, is_leaf=False) * output)
        if(self.b.requires_grad):
            self.b.backward(Tensor(self.b.arr.ones_like(self.b.item), device=self.b.device, is_leaf=False) * output)

class SubBackward(Functional):
    # a - b
    def __init__(self, a: Tensor, b: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
        self.b = b
    
    def backward(self, output = Tensor([1])):
        output.to(self.a.device)
        if(self.a.requires_grad):
            self.a.backward(Tensor(self.a.arr.ones_like(self.a.item), device=self.a.device, is_leaf=False) * output)
        if(self.b.requires_grad):
            self.b.backward(-Tensor(self.b.arr.ones_like(self.b.item), device=self.b.device, is_leaf=False) * output)

class MulBackward(Functional):
    # a * b
    def __init__(self, a: Tensor, b: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
        self.b = b
    
    def backward(self, output = Tensor([1])):
        output.to(self.a.device)
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
        output.to(self.a.device)
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
        output.to(self.a.device)
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
        output.to(self.a.device)
        if(self.a.requires_grad):
            self.a.backward(self.b * self.a**(self.b-1) * output)

class ExpBackward(Functional):
    # exp(a)
    def __init__(self, a: Tensor, requires_grad=False):
        super().__init__()
        self.a = a
    
    def backward(self, output = Tensor([1])):
        output.to(self.a.device)
        if(self.a.requires_grad):
            self.a.backward(Tensor(self.a.arr.exp(self.a.item), device=self.b.device, is_leaf=False) * output)

class ReluBackward(Functional):
    def __init__(self, x: Tensor, requires_grad=False):
        super().__init__()
        self.x = x
    
    def backward(self, output = Tensor([1])):
        output.to(self.x.device)
        if(self.x.requires_grad):
            self.x.backward((self.x > 0) * output)

class SoftmaxBackward(Functional):
    def __init__(self, x: Tensor, axis: int, requires_grad=False):
        super().__init__()
        self.x = x
        self.axis = axis
    
    def backward(self, output = Tensor([1])):
        output.to(self.x.device)
        def softmax(x, axis):
            phi = x.max(axis=axis)
            e = self.x.arr.exp(x - phi)
            return e / e.sum(axis=axis)

        s = softmax(self.x.item, self.axis)
        res = Tensor(s*(s-1), device=self.x.device, requires_grad=False, is_leaf=False)
        if(self.x.requires_grad):
            self.x.backward(res * output)

class FlattenBackward(Functional):
    def __init__(self, x: Tensor, requires_grad=False):
        super().__init__()
        self.x = x
    
    def backward(self, output = Tensor([1])):
        output.to(self.x.device)
        if(self.x.requires_grad):
            res = Tensor(self.x.item.T.reshape(self.x.shape, order='C'), device=self.x.device, is_leaf=False)
            self.x.backward(res)

class Cov2dBackward(Functional):
    def __init__(self, x: Tensor, w: Tensor, requires_grad=False):
        super().__init__()
        self.x = x
        self.w = w
    
    def backward(self, output = Tensor([[[[1]]]])):
        output.to(self.x.device)
        _, in_c, w_size, _ = self.w.shape
        batch_size, x_c, x_size, _ = self.x.shape
        batch_size, out_c, z_size, _ = output.shape
        steps = x_size - z_size + 1

        # if(self.x.requires_grad):
        #     dx = self.x.arr.zeros([batch_size, in_c, x_size, x_size])
        #     for i in range(z_size):
        #         for j in range(z_size):
        #             dx[:, :, i:i+w_size, j:j+w_size] += (self.w.item[:,None,:,:,:] * output.item[:, :, i, j].transpose(1,0)[:,:,None,None,None]).sum(axis=0)

        #     self.x.backward(Tensor(dx, device=self.x.device, is_leaf=False))

        if(self.w.requires_grad):            
            _z = output.item.transpose(1,0,2,3).reshape([out_c, -1], order='C')
            _x = self.x.arr.zeros([batch_size*z_size**2, steps**2*x_c])
            for i in range(steps):
                for j in range(steps):
                    loc = i*steps+j
                    _x[:,loc::steps**2] =  self.x.item[:, :, i:i+z_size, j:j+z_size].transpose(1,0,2,3).reshape([x_c,-1]).T   # ?????????cov2d, ??????im2col??????
            dw = (_z @ _x).reshape(out_c,x_c,steps,steps) / batch_size

            self.w.backward(Tensor(dw, device=self.x.device, is_leaf=False))

class MaxPoolingBackward(Functional):
    def __init__(self, x: Tensor, p_size:int, argmax: Tensor, requires_grad=False):
        super().__init__()
        self.x = x
        self.p_size = p_size
        self.argmax = argmax
    
    def backward(self, output = Tensor([[[[1]]]])):
        output.to(self.x.device)
        batch_size, c, z_size, _ = output.shape
        x_size = z_size * self.p_size
        dx = self.x.arr.zeros([batch_size*c*z_size*z_size, self.p_size*self.p_size])
        dx[self.x.arr.arange(self.argmax.item.size), self.argmax.item] = output.item.flatten()
        dx = dx.reshape([batch_size,c,z_size,z_size,self.p_size,self.p_size], order='C').transpose(0,1,4,5,2,3)
        dxx = self.x.arr.zeros([batch_size, c, x_size, x_size])
        
        for i in range(self.p_size):
            for j in range(self.p_size):
                dxx[:,:,i::self.p_size,j::self.p_size] = dx[:, :, i, j,:,:]
        
        self.x.backward(Tensor(dxx, device=self.x.device, is_leaf=False))

class CrossEntropyBackward(Functional):
    def __init__(self, x: Tensor, y: Tensor, requires_grad=False):
        super().__init__()
        self.x = x
        self.y = y
    
    def backward(self, output = Tensor([1])):
        output.to(self.x.device)
        def softmax(x):
            phi = x.max(axis=0)
            e = self.x.arr.exp(x - phi)
            return e / e.sum(axis=0)
        a = softmax(self.x.item)
        res = Tensor(a - self.y.item, device=self.x.device, requires_grad=False, is_leaf=False)
        if(self.x.requires_grad):
            self.x.backward(res * output)