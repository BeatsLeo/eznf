class Module:
    def __init__(self):
        self.networks = None
        self.w = None

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError('forward() is not implemented')

    def parameters(self):
        if(self.networks):
            for network in self.networks:
                if(network.w):
                    yield network.w