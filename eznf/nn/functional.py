from eznf.tensor.tensor import Tensor

def istensor(input):
    if(isinstance(input,Tensor)!=True):
        raise ValueError('not tensor')
    

def linear(input, weight, bias=None): 
    istensor(input)
    return input @ weight.T + bias


def relu(input) :
    istensor(input)
    return input * (input > 0)


def sigmoid(input) :
    istensor(input)
    return 1.0/(1.0+Tensor.exp(-input))


def tanh(input) :
    istensor(input)
    return (Tensor.exp(input)-Tensor.exp(-input))/(Tensor.exp(input)+Tensor.exp(-input))

def softmax(input):      
    istensor(input)                      
    exp_input=Tensor.exp(input)
    exp_sum = sum(exp_input)
    return exp_input/exp_sum

def mes_loss(input,y):  
    istensor(input)
    istensor(y)
    output=0
    for i in range(len(input)):
        output += (input[i]-y[i])**2/2
    return output

def cross_entropy(input,y):  
    istensor(input)
    istensor(y)
    input=Tensor.log(input)
    output=0
    for i in range(len(input)):
        output += -y[i]*input[i]
    return output
