import numpy as np
import eznf

def from_numpy(arr: np.ndarray):
    # 建议参数类型： np.ndarray，返回值类型:eznf.Tensor
    if(type(arr) is np.ndarray):
        return eznf.Tensor(arr)
    else:
        print("Type error, arg 's type isn't ndarray")

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
    #若arr为空，返回True
    return arr.size == 0

def to_numpy(arr):
    #将eznf.Tensor类型变量转化为numpy
    if (type(arr) is eznf.Tensor):
        return arr.item
    else:
        print("Type error, arg 's type isn't eznf.Tensor")

def to_list(arr):
    #将eznf.Tensor类型变量转化为list
    return to_numpy(arr).tolist()