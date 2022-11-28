<<<<<<< HEAD
# import eznf
<<<<<<< Updated upstream
from cmath import tan
=======
# import eznf.utils.utils as utils
>>>>>>> Stashed changes
import torch
import numpy as np


<<<<<<< Updated upstream


#支持gpu暂且没写，而且只用了pytorch里面的tensor
a=torch.tensor([[1,1],[1,1],[1,1]])
b=np.array([1,1])
=======
#支持gpu暂且没写，而且只用了pytorch里面的tensor
a=torch.tensor([[[[1,1],[1,1]],[[2,2],[2,2]]],[[[1,1],[1,1]],[[2,2],[2,2]]]])
b=np.array([[[[1,1,1],[1,1,1]],[[2,2,2],[2,2,2]]],[[[1,1,1],[1,1,1]],[[2,2,2],[2,2,2]]]])
b=np.transpose(b,(2,3,1,0))
print(b.shape)
>>>>>>> Stashed changes

=======
import eznf
from eznf.autograd import function
import numpy as np

>>>>>>> 04f12272c7591d1117cb9947e213eed8466d6a5a
def istensor(input):
    if(isinstance(input, eznf.Tensor)!=True):
        raise ValueError('not tensor')

def relu(x):
    if(x.requires_grad):
        grad_fn = function.ReluBackward(x=x, requires_grad=x.requires_grad)
    else:
        grad_fn = None
    res = x.item * (x.item > 0)
    return eznf.Tensor(res, requires_grad=x.requires_grad, grad_fn=grad_fn, is_leaf=False)

def linear(x, w):
    return w @ x

def sigmoid(input) :
    istensor(input)
    return 1.0/(1.0+eznf.exp(-input))

def tanh(input) :
    istensor(input)
<<<<<<< HEAD
<<<<<<< Updated upstream
    shape = input.shape
    input=torch.reshape(input,(-1,1))
    for i in range(len(input)):
        input[i]=max(0,input[i])
    output=torch.reshape(input,shape)
=======
    input=np.array(input)
    # print(type(input.shape))
    # np_zero = np.zeros(list(input.shape))
    output = np.maximum(0,input)
    # output = np.maximum(np_zero,input)
    output=torch.tensor(output)
>>>>>>> Stashed changes
=======
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

# 最大池化, stride = p_size, padding = 0
def max_pooling(x, p_size):
    """
        x: batch_size * channels * w * h
    """    
    batch_size, c, x_size, _ = x.shape

    if(int(x_size/p_size) != x_size/p_size):
        raise ValueError('x_size must be divided exactly by p_size')

    steps = x_size // p_size
    _z = np.zeros([p_size**2, batch_size*c*steps**2])

    for i in range(p_size):
        for j in range(p_size):
            _z[i*p_size+j, :] = x.item[:, :, i::p_size, j::p_size].reshape([-1], order='C')    # 将输入展平(im2col)

    z = _z.max(axis=0).reshape([batch_size, c, steps, steps])
    arg_max = _z.argmax(axis=0)

    if(x.requires_grad):
        grad_fn = function.MaxPoolingBackward(x, p_size=p_size, argmax=eznf.Tensor(arg_max, is_leaf=False), requires_grad=x.requires_grad)
    else:
        grad_fn = None

    return eznf.Tensor(z, requires_grad=x.requires_grad, grad_fn=grad_fn, is_leaf=False)

def mse(input,y):  
    istensor(input)
    istensor(y)
    output=0
    for i in range(len(input)):
        output += (input[i]-y[i])**2/2
>>>>>>> 04f12272c7591d1117cb9947e213eed8466d6a5a
    return output

# 交叉熵损失   
def cross_entropy(x, y):
    if(x.requires_grad):
        grad_fn = function.CrossEntropyBackward(x=x, y=y, requires_grad=x.requires_grad)
    else:
        grad_fn = None

<<<<<<< HEAD
<<<<<<< Updated upstream
def softmax(input, dim=None)-> torch.Tensor:        #stacklevel=3, dtype=None 没管,dim写不来
=======
def softmax(input)-> torch.Tensor:      
>>>>>>> Stashed changes
    istensor(input)                        
    exp_input=torch.tensor(input)
    exp_sum = torch.sum(exp_input)
    return exp_input/exp_sum


<<<<<<< Updated upstream
# def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> torch.Tensor: 
    
    
=======
# def conv1d(input,input_channel, output_channel,kernal_size, stride=1, padding=0) -> torch.Tensor: 
# def conv2d(input,input_channel, output_channel,kernal_size, stride=1, padding=0) -> torch.Tensor: 
# def conv3d(input,input_channel, output_channel,kernal_size, stride=1, padding=0) -> torch.Tensor: 



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




def mes_loss(input,y)-> torch.Tensor:  
    istensor(input)
    istensor(y)
    output=0
    for i in range(len(input)):
        output += (input[i]-y[i])**2/2
    return output

def cross_entropy(input,y)-> torch.Tensor:  
    istensor(input)
    istensor(y)
    input=softmax(input)
    input=torch.log(input)
    output=0
    for i in range(len(input)):
        output += -y[i]*input[i]
    return output



>>>>>>> Stashed changes
=======
    a = softmax(x, 0)
    res = -(y.item*np.log(a.item)).sum()
    return eznf.Tensor(res, requires_grad=x.requires_grad, grad_fn=grad_fn, is_leaf=False)
>>>>>>> 04f12272c7591d1117cb9947e213eed8466d6a5a
