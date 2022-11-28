<<<<<<< HEAD
import eznf.tensor.tensor as tensor
import numpy as np
# import torch

def istensor(input):
    if(isinstance(input,tensor.Tensor)!=True):
        raise ValueError('not tensor')

def from_numpy(arr: np.ndarray) -> tensor.Tensor:
    # 建议参数类型： np.ndarray，返回值类型:eznf.tensor
=======
import numpy as np
import eznf

def from_numpy(arr: np.ndarray):
    # 建议参数类型： np.ndarray，返回值类型:eznf.Tensor
>>>>>>> 04f12272c7591d1117cb9947e213eed8466d6a5a
    if(type(arr) is np.ndarray):
        return tensor.Tensor(arr)
    else:
        raise ValueError("Type error, arg 's type isn't ndarray")

<<<<<<< HEAD
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
=======
def ones(*args):
    #创建一个全是1的 eznf.Tensor类型
    if(isinstance(args[0], list)):
        return eznf.Tensor(np.ones(*args))
    else:
        return eznf.Tensor(np.ones(args))

def zeros(*args):
    #创建一个全是0的 eznf.Tensor类型
    if(isinstance(args[0], list)):
        return eznf.Tensor(np.zeros(*args))
    else:
        return eznf.Tensor(np.zeros(args))

def randn(*args):
    return eznf.Tensor(np.random.randn(*args))

def empty(arr):
>>>>>>> 04f12272c7591d1117cb9947e213eed8466d6a5a
    #若arr为空，返回True
    return arr.size == 0

<<<<<<< HEAD
def to_numpy(arr: tensor.Tensor) -> np.ndarray:
    #将eznf.tensor类型变量转化为numpy
    if (type(arr) is tensor.Tensor):
        return np.array(arr)
    else:
        raise ValueError("Type error, arg 's type isn't eznf.tensor")

def to_list(arr: tensor.Tensor) -> list:
    #将eznf.tensor类型变量转化为list
=======
def to_numpy(arr):
    #将eznf.Tensor类型变量转化为numpy
    if (type(arr) is eznf.Tensor):
        return arr.item
    else:
        print("Type error, arg 's type isn't eznf.Tensor")

def to_list(arr):
    #将eznf.Tensor类型变量转化为list
>>>>>>> 04f12272c7591d1117cb9947e213eed8466d6a5a
    return to_numpy(arr).tolist()

def one_hot(arr, length):
    if(not isinstance(arr, eznf.Tensor)):
        raise TypeError('array must be a Tensor')
    if(len(arr.shape) != 1):
        raise ValueError('array should have 1 dimensions')

    num = arr.size()
    oh = np.zeros([num, length])
    oh[np.arange(num), arr.item] = 1

    return eznf.Tensor(oh, requires_grad=False, is_leaf=False)