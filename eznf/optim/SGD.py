import eznf

class SGD:
    def __init__(self, alpha:float, model:eznf.nn.Module):
        self.alpha = alpha
        self.m = model

    def step(self):
        for w in self.m.parameters():
            w.item = w.item - self.alpha * w.grad.item
            
    def zero_grad(self):
        for w in self.m.parameters():
            w.grad = None