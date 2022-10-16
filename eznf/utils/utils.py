import eznf
import numpy as np

def from_numpy(arr: np.ndarray):
    return eznf.Tensor(arr)