# import eznf
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
    input=np.array(input)
    output = np.maximum(0,input)
    output=torch.tensor(output)
    return output
        
def sigmoid(input) -> torch.Tensor:
    istensor(input)
    return 1.0/(1.0+torch.exp(-input))

def tanh(input) -> torch.Tensor:
    istensor(input)
    return (torch.exp(input)-torch.exp(-input))/(torch.exp(input)+torch.exp(-input))

def softmax(input)-> torch.Tensor:      
    istensor(input)                        
    exp_input=torch.tensor(input)
    exp_sum = torch.sum(exp_input)
    return exp_input/exp_sum




# def conv1d(input, sad, bias=None, stride=1, padding=0) -> torch.Tensor: 
    
# def conv2d(input, weight, bias=None, stride=1, padding=0) -> torch.Tensor: 
# def conv3d(input, weight, bias=None, stride=1, padding=0) -> torch.Tensor: 
# def conv2d(input, weight, bias=None, stride=1, padding=0) -> torch.Tensor: 
# def conv3d(input, weight, bias=None, stride=1, padding=0) -> torch.Tensor: 
# def conv2d(input, weight, bias=None, stride=1, padding=0) -> torch.Tensor: 
# def conv3d(input, weight, bias=None, stride=1, padding=0) -> torch.Tensor: 
# def conv2d(input, weight, bias=None, stride=1, padding=0) -> torch.Tensor: 
# def conv3d(input, weight, bias=None, stride=1, padding=0) -> torch.Tensor: 


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



