import eznf.tensor.tensor as tensor
import eznf.nn.functional as F
import eznf
from .module import Module



class Perceptron(Module):
    def __init__(self, learning_rate, max_iter=1000):
        super().__init__()
        self.learning_rate = learning_rate  # 学习率
        self.max_iter = max_iter            # 最大迭代次数
        # 模型参数
        self.weight = None
        self.bias = None
        # 数据输入
        # self.train_X = train_X
        # self.train_Y = train_Y
    
    def __call__(self, learning_rate, max_iter):
        # self.fit(self, self.train_X, self.train_Y)
        pass

    def fit(self, X, y):
        n, m = X.item.shape[0],X.item.shape[1]
        # print(n,m)
        # 初始化模型参数
        # self.weight = utils.ones(m)
        self.weight = eznf.ones(m)
        # self.bias = utils.zeros(1)
        self.bias = eznf.zeros(1)

        for i in range(self.max_iter):
            # 标记本轮计算是否存在分类错误
            has_error = 0
            # 遍历训练集
            for idx in range(n):
                Xi = X[idx, :]
                yi = y[idx]
                # 计算线性函数输出值
                out = (self.weight.T @ Xi) + self.bias
                # 分类错误则更新
                if out * yi <= 0:
                    # 标记本轮循环遇到了错误样本
                    has_error = 1   
                    #weigh和bias的更新
                    self.bias += self.learning_rate * yi
                    self.weight += self.learning_rate * (yi * Xi)
                    
            if has_error == 0:
                # 本轮迭代所有样本都分类正确，终止循环
                break

    def predict(self, X):
        # 每个样本计算出的函数值
        f_value = (self.weight @ X) + self.bias
        # 计算对应的符号函数值，正数为1，负数为-1，0为0
        pred = F.relu(f_value)
        # pred[pred >= 0] = 1
        if(pred > 0):
            return eznf.ones(1)
        else:
            return tensor.Tensor([-1])
