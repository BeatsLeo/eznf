from eznf import Tensor

class Functional():
    def __init__(self, requires_grad=False):
        self.requires_grad = requires_grad
        self.next_functions = []
        self.saved = None
    
    def forward(self, *args):
        raise NotImplementedError("You must implement the forward function for custom autograd.Function.")

    def backward(self, *grad_outputs):
        raise NotImplementedError("You must implement either the backward method for your custom autograd.Function to use it with backward mode AD.")

    def save_for_backward(self, result: Tensor):
        self.saved = result.copy()

    def saved_tensors(self):
        return self.saved