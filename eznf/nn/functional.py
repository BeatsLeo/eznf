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


def mes_loss(input,y):  
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

    a = softmax(x.item)
    res = -(y*np.log(a)).sum()
    return eznf.Tensor(res, requires_grad=x.requires_grad, grad_fn=grad_fn, is_leaf=False)
