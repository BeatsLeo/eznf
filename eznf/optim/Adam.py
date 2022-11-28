from eznf import Tensor
import eznf 

class Adam:
    def __init__(self, model:eznf.nn.Module, alpha:float, belta1:float = 0.9, belta2:float = 0.999, epsilon:float = 1e-8):
        self.model = model
        self.alpha = alpha
        self.belta1 = belta1
        self.belta2 = belta2
        self.epsilon = epsilon
        
        self.t = 0

        self.m = {}
        #self.m[self.t] =
        self.v = {}
        #self.v{self.t} = 
        
    def step(self):
        self.t += 1

        #gt = w.grad.item
        #self.m[self.t] = self.belta1 * self.m[self.t - 1] + (1 - self.belta1) * gt
        #self.v[self.t] = self.belta2 * self.v[self.t - 1] + (1 - self.belta2) * (gt ** 2)
        mt_ = self.m[self.t] / (1 - self.belta1)
        vt_ = self.v[self.t] / (1 - self.belta2)
        #w = w - self.alpha * mt_/((vt_**0.5) + self.epsilon) 

    def zero_grad(self):
        for w in self.m.parameters():
            w.grad = None
