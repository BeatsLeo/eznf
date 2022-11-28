import eznf
from eznf.autograd import function
import numpy as np

# 线性函数
def linear(x, w):
    return w @ x

# relu激活函数
def relu(x):
    if(x.requires_grad):
        grad_fn = function.ReluBackward(x=x, requires_grad=x.requires_grad)
    else:
        grad_fn = None
    res = x.item * (x.item > 0)
    return eznf.Tensor(res, device=x.device, requires_grad=x.requires_grad, grad_fn=grad_fn, is_leaf=False)

# sigmoid激活函数
def sigmoid(x):
    return 1 / (1 + (-x).exp())

# tanh激活函数
def tanh(x) :
    return (x.exp()- (-x).exp()) / (x.exp() + (-x).exp())

# softmax激活函数
def softmax(x, axis):
    if(x.requires_grad):
        grad_fn = function.SoftmaxBackward(x=x, axis=axis, requires_grad=x.requires_grad)
    else:
        grad_fn = None

    phi = x.item.max(axis=axis)
    e = x.arr.exp(x.item - phi)
    res = e / e.sum(axis=axis)
    return eznf.Tensor(res, device=x.device, requires_grad=x.requires_grad, grad_fn=grad_fn, is_leaf=False)

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
    _z = x.arr.zeros([in_channels*w_size**2, batch_size*steps**2]) # 为将输入展平做准备

    for i in range(w_size):
        for j in range(w_size):
            _z[i*w_size+j::w_size**2, :] = x.item[:, :, i:steps+i, j:steps+j].transpose(1,0,2,3).reshape([in_channels,-1], order='C')    # 将输入展平(im2col)

    z = (_w @ _z).reshape([out_channels, batch_size, steps, steps], order='C').transpose(1,0,2,3)   # 矩阵相乘后还原(卷积)
    return eznf.Tensor(z, device=x.device, requires_grad=requires_grad, grad_fn=grad_fn, is_leaf=False)

# 最大池化, stride = p_size, padding = 0
def max_pooling(x, p_size):
    """
        x: batch_size * channels * w * h
    """    
    batch_size, c, x_size, _ = x.shape

    if(int(x_size/p_size) != x_size/p_size):
        raise ValueError('x_size must be divided exactly by p_size')

    steps = x_size // p_size
    _z = x.arr.zeros([p_size**2, batch_size*c*steps**2])

    for i in range(p_size):
        for j in range(p_size):
            _z[i*p_size+j, :] = x.item[:, :, i::p_size, j::p_size].reshape([-1], order='C')    # 将输入展平(im2col)

    z = _z.max(axis=0).reshape([batch_size, c, steps, steps])
    arg_max = _z.argmax(axis=0)

    if(x.requires_grad):
        grad_fn = function.MaxPoolingBackward(x, p_size=p_size, argmax=eznf.Tensor(arg_max, device=x.device, is_leaf=False), requires_grad=x.requires_grad)
    else:
        grad_fn = None

    return eznf.Tensor(z, device=x.device, requires_grad=x.requires_grad, grad_fn=grad_fn, is_leaf=False)

# 均方误差损失
def mse(x, y):  
    return ((x - y)**2).sum() / 2

# 交叉熵损失   
def cross_entropy(x, y):
    if(x.requires_grad):
        grad_fn = function.CrossEntropyBackward(x=x, y=y, requires_grad=x.requires_grad)
    else:
        grad_fn = None

    a = softmax(x, 0)
    res = -(y.item*x.arr.log(a.item)).sum()
    return eznf.Tensor(res, device=x.device, requires_grad=x.requires_grad, grad_fn=grad_fn, is_leaf=False)