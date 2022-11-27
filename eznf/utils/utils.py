import eznf.tensor.tensor as tensor
import numpy as np
# import torch

def istensor(input):
    if(isinstance(input,tensor.Tensor)!=True):
        raise ValueError('not tensor')

def from_numpy(arr: np.ndarray) -> tensor.Tensor:
    # 建议参数类型： np.ndarray，返回值类型:eznf.tensor
    if(type(arr) is np.ndarray):
        return tensor.Tensor(arr)
    else:
        raise ValueError("Type error, arg 's type isn't ndarray")

def ones(shape: int or list) -> tensor.Tensor:
    #创建一个全是1的 eznf.tensor类型
    if(type(shape) is int or list):
        onesTensor = from_numpy(np.ones(shape))
        return onesTensor
    else:
        raise ValueError("Type error, arg 's type isn't int nor list")

def zeros(shape: int or list) -> tensor.Tensor:
    #创建一个全是0的 eznf.tensor类型
    if(type(shape) is int or list):
        zerosTensor = from_numpy(np.zeros(shape))
        return zerosTensor
    else:
        raise ValueError("Type error, arg 's type isn't int nor list")

def empty(arr: tensor.Tensor) -> bool:
    #若arr为空，返回True
    return len(arr) == 0

def to_numpy(arr: tensor.Tensor) -> np.ndarray:
    #将eznf.tensor类型变量转化为numpy
    if (type(arr) is tensor.Tensor):
        return np.array(arr)
    else:
        raise ValueError("Type error, arg 's type isn't eznf.tensor")

def to_list(arr: tensor.Tensor) -> list:
    #将eznf.tensor类型变量转化为list
    return to_numpy(arr).tolist()
