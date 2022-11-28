import eznf
import eznf.tensor.tensor as tensor
import eznf.nn.functional as F
from .module import Module

class Hebb(Module):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate  # 学习率
        # 模型参数
        self.weight = None

    def fit(self, train_X,train_Y):
        n, m = train_X.item.shape[0],train_X.item.shape[1]
        self.weight = eznf.ones(m)
        for idx in range(n):
            Xi = train_X[idx, :]
            yi = train_Y[idx]
            # sum = (self.weight.T @ Xi) - yi
            sum = (self.weight.T @ Xi) + tensor.Tensor([0])
            # out = F.relu(sum)
            # out = 1 if(sum > 0) else -1
            # print(type(out))
            if sum * yi > 0:
                out = 1
            else:
                out = -1
            # print(yi)
            # out = -1
            delta_w = self.learning_rate * out * Xi
            self.weight += delta_w

    def forward(self, test_X):
        sum = (self.weight.T @ test_X)
        # out = 1 if(sum > 0) else -1
        # print(sum)
        out = tensor.Tensor([-1])
        return out
