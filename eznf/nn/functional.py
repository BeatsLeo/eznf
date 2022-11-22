# import eznf
from cmath import tan
import torch
import numpy as np




#支持gpu暂且没写，而且只用了pytorch里面的tensor
a=torch.tensor([[1,1],[1,1],[1,1]])
b=np.array([1,1])

def istensor(input):
    if(isinstance(input,torch.Tensor)!=True):
        raise ValueError('not tensor')
    

def linear(input, weight, bias=None)-> torch.Tensor: 
    istensor(input)
    return input @ weight.T + bias


def relu(input) -> torch.Tensor:
    istensor(input)
    shape = input.shape
    input=torch.reshape(input,(-1,1))
    for i in range(len(input)):
        input[i]=max(0,input[i])
    output=torch.reshape(input,shape)
    return output
        
def sigmoid(input) -> torch.Tensor:
    istensor(input)
    return 1.0/(1.0+torch.exp(-input))

def tanh(input) -> torch.Tensor:
    istensor(input)
    return (torch.exp(input)-torch.exp(-input))/(torch.exp(input)+torch.exp(-input))

def softmax(input, dim=None)-> torch.Tensor:        #stacklevel=3, dtype=None 没管,dim写不来
    istensor(input)                        
    exp_input=torch.tensor(input)
    exp_sum = torch.sum(exp_input)
    return exp_input/exp_sum


# def conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> torch.Tensor: 
    
    