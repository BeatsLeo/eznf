from .module import Module
import numpy as np

class MSE(Module):
    def __init__(self):
        pass

    def __call__(self, *args):
        self.forward(args)

    def forward(self, *args):
        pass

# class CrossEntropyLoss(Module):
#     def __init__(self, batch_size):
#         self.a = None
#         self.y = None
#         self.batch_size = batch_size

#     def __call__(self, x: np.ndarray, y: np.ndarray):
#         self.y = y
#         self.a = softmax(x)
#         return cross_entropy(self.a, y, self.batch_size)

#     def backward(self):
#         return dcross_entropy(self.a, self.y)