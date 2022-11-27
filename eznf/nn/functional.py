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

def istensor(input):
    if(isinstance(input,torch.Tensor)!=True):
        raise ValueError('not tensor')
    

def linear(input, weight, bias=None)-> torch.Tensor: 
    istensor(input)
    return input @ weight.T + bias


def relu(input) -> torch.Tensor:
    istensor(input)
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
    return output
        
def sigmoid(input) -> torch.Tensor:
    istensor(input)
    return 1.0/(1.0+torch.exp(-input))

def tanh(input) -> torch.Tensor:
    istensor(input)
    return (torch.exp(input)-torch.exp(-input))/(torch.exp(input)+torch.exp(-input))

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
