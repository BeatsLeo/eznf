import eznf
from eznf.autograd import function
import numpy as np

def istensor(input):
    if(isinstance(input, eznf.Tensor)!=True):
        raise ValueError('not tensor')

def relu(x) :
    if(x.requires_grad):
        grad_fn = function.ReluBackward(x=x, requires_grad=x.requires_grad)
    else:
        grad_fn = None
    res = x.item * (x.item > 0)
    return eznf.Tensor(res, requires_grad=x.requires_grad, grad_fn=grad_fn, is_leaf=False)
  
def sigmoid(input) :
    istensor(input)
    return 1.0/(1.0+eznf.exp(-input))

def tanh(input) :
    istensor(input)
    return (eznf.exp(input)-eznf.exp(-input))/(eznf.exp(input)+eznf.exp(-input))

# softmax激活函数
def softmax(x, axis):
    if(x.requires_grad):
        grad_fn = function.SoftmaxBackward(x=x, axis=axis, requires_grad=x.requires_grad)
    else:
        grad_fn = None

    phi = x.item.max(axis=axis)
    e = np.exp(x.item - phi)
    res = e / e.sum(axis=axis)
    return eznf.Tensor(res, requires_grad=x.requires_grad, grad_fn=grad_fn, is_leaf=False)

# 卷积, stride = 1, padding = 0
def cov2d(x, w):
    """
        x: batch_size * channels * w * h
        w: out_channels * channels * w * h
    """    
    requires_grad = x.requires_grad or w.requires_grad
    if(requires_grad):
        grad_fn = function.Cov2dBackward(x, w, requires_grad=requires_grad)
    else:
        grad_fn = None

    batch_size, in_channels, x_size, _ = x.shape
    out_channels, _, w_size, _ = w.shape
    steps = x_size - w_size + 1

    _w = w.item.reshape([out_channels, -1], order='C')   # 将卷积核展平
    _z = np.zeros([in_channels*w_size**2, batch_size*steps**2]) # 为将输入展平做准备

    for i in range(w_size):
        for j in range(w_size):
            _z[i*w_size+j::w_size**2, :] = x.item[:, :, i:steps+i, j:steps+j].transpose(1,0,2,3).reshape([in_channels,-1], order='C')    # 将输入展平(im2col)

    z = (_w @ _z).reshape([out_channels, batch_size, steps, steps], order='C').transpose(1,0,2,3)   # 矩阵相乘后还原(卷积)
    return eznf.Tensor(z, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

def mse(input,y):  
    istensor(input)
    istensor(y)
    output=0
    for i in range(len(input)):
        output += (input[i]-y[i])**2/2
    return output

# 交叉熵损失   
def cross_entropy(x, y):
    if(x.requires_grad):
        grad_fn = function.CrossEntropyBackward(x=x, y=y, requires_grad=x.requires_grad)
    else:
        grad_fn = None

    a = softmax(x, 0)
    res = -(y.item*np.log(a.item)).sum()
    return eznf.Tensor(res, requires_grad=x.requires_grad, grad_fn=grad_fn, is_leaf=False)