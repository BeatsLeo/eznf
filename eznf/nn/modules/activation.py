from .module import Module

class ReLU(Module):
    def __init__(self):
        pass

    def __call__(self, *args):
        self.forward(args)

    def forward(self, *args):
        pass