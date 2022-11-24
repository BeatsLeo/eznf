import numpy as np

class Tensor:
    def __init__(self, *args, device=None):
        if(isinstance(args[0], list) or isinstance(args[0], np.ndarray)):
            self.item = np.array(args[0], float)
        else:
            self.item = np.random.random(args)

        self.device = device


    def __str__(self):
        return 'tensor({})'.format(self.item)

    def __repr__(self):
        return 'tensor({})'.format(self.item)
