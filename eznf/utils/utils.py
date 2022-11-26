import eznf
import numpy as np
import torch

def from_numpy(arr: np.ndarray) -> eznf.tensor:
    # 建议参数类型： np.ndarray，返回值类型:eznf.tensor
    if(type(arr) is np.ndarray):
        return eznf.Tensor(arr)
    else:
        print("Type error, arg 's type isn't ndarray")

def ones(shape: int or list) -> eznf.tensor:
    #创建一个全是1的 eznf.tensor类型
    if(type(shape) is int or list):
        onesTensor = from_numpy(np.ones(shape))
        return onesTensor
    else:
        print("Type error, arg 's type isn't int nor list")

def zeros(shape: int or list) -> eznf.tensor:
    #创建一个全是0的 eznf.tensor类型
    if(type(shape) is int or list):
        zerosTensor = from_numpy(np.zeros(shape))
        return zerosTensor
    else:
        print("Type error, arg 's type isn't int nor list")

def empty(arr: eznf.tensor) -> bool:
    #若arr为空，返回True
    return len(arr) == 0

def to_numpy(arr: eznf.tensor) -> np.ndarray:
    #将eznf.tensor类型变量转化为numpy
    if (type(arr) is eznf.tensor):
        return torch.tensor(arr)
    else:
        print("Type error, arg 's type isn't eznf.tensor")

def to_list(arr: eznf.tensor) -> list:
    #将eznf.tensor类型变量转化为list
    return to_numpy(arr).tolist()
