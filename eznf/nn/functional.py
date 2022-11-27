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


def conv2d(x,input_channel,output_channel,kernal_size):       #默认无padding，移动1
    #x  [width,height,channel,batch]                                               
    kernal = np.random.randn(output_channel,kernal_size,kernal_size) 
    width,height,batch=x.shape[0],x.shape[1],x.shape[3]           
    
    out=np.zeros([width-kernal_size+1,height-kernal_size+1,output_channel,batch])
    for k in range(batch):
        for i in range(output_channel):
            for j in range(input_channel):
                for a in range(width-kernal_size+1):
                    for b in range(height-kernal_size+1):
                        out[a,b,i,k]+=np.sum(np.sum(x[a:a+kernal_size,b:b+kernal_size,j,k]*kernal[i]))
                    if b==width-kernal_size and a!=height-kernal_size:
                        b=0
    return out,kernal


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
