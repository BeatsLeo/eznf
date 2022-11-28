import eznf.utils.utils as utils
# import eznf.tensor.tensor as tensor
import eznf.nn.functional as F
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
        self.weight = utils.ones(m)
        self.bias = utils.zeros(1)
        # 将输入的ndarray转换为tensor
        # X = utils.from_numpy(X).float()
        # y = utils.from_numpy(y).float()

        for i in range(self.max_iter):
            # 标记本轮计算是否存在分类错误
            has_error = 0
            # 遍历训练集
            for idx in range(n):
                Xi = X[idx, :]
                yi = y[idx]
                # 计算线性函数输出值
                out = (self.weight.T @ Xi) + self.bias
                # out = self.weight.mul(Xi)
                # 分类错误则更新
                # print(out * yi)
                if out * yi <= 0:
                    # 标记本轮循环遇到了错误样本
                    has_error = 1   
                    #weigh和bias的更新
                    self.weight += self.learning_rate * yi * Xi
                    self.bias += self.learning_rate * yi
            if has_error == 0:
                # 本轮迭代所有样本都分类正确，终止循环
                break

    def predict(self, X, Y):
        # 每个样本计算出的函数值
        f_value = (self.weight.T @ X) + self.bias
        # print(f_value)
        # print(type(f_value))
        # 计算对应的符号函数值，正数为1，负数和0为-1
        pred = F.relu(f_value)
        # pred = f_value[0]
        pred[pred == 0] = 1
<<<<<<< HEAD
<<<<<<< Updated upstream
        return pred
=======
        # pred = -1 if pred < 0 else 1
        return pred
>>>>>>> Stashed changes
=======
        return pred
>>>>>>> 04f12272c7591d1117cb9947e213eed8466d6a5a
