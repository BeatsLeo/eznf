class Module:
    def __init__(self, device='cpu'):
        self.device = device
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
    
    def to(self, device):
        if(self.networks):
            for network in self.networks:
                if(network.w):
                    network.w.to(device)
                    if(network.w.device == 'gpu'):
                        network.w.device = 'gpu'
                    else:
                        network.w.device = 'cpu'
        else:
            self.w.to(device)
            if(self.w.device == 'gpu'):
                self.w.device = 'gpu'
            else:
                self.w.device = 'cpu'

        return self