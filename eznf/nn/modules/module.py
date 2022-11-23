class Module:
    def __init__(self):
        pass

    def __call__(self, *args):
        self.forward(args)

    def forward(self, *args):
        raise NotImplementedError('forward() is not implemented')

    def backward(self, *args):
        pass