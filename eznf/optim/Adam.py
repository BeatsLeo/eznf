import eznf 
import numpy as np
from eznf import Tensor

class Adam:
    def __init__(self, lr, model: eznf.nn.Module):
        self.lr = lr
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-7
        self.model = model
        self.m = None
        self.v = None
        self.step = 0
 
    def step(self):
        if self.m is None:
            self.m = {}
            self.v = {}

            for w in self.model.parameters():
                self.m[w] = np.zeros_like(w)
                self.v[w] = np.zeros_like(w)
 
        self.step += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.step) / (1.0 - self.beta1**self.step)
        for w in self.model.parameters():
            self.m[w] = (self.beta1 * self.m[w] + (1 - self.beta1) * w.grad.item)
            self.v[w] = (self.beta2 * self.v[w] + (1 - self.beta2) * w.grad.item**2)
            w.item -= lr_t * self.m[w] / (np.sqrt(self.v[w]) + self.eps)


    def zero_grad(self):
        for w in self.m.parameters():
            w.grad = None
