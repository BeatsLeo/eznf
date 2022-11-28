import matplotlib.pyplot as plt
import numpy as np
class VTrain(object):
    def __init__(self, model, x_train, x_test, y_train, y_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        
    def test(self):
        print('???')

    def vtrain(self, alpha, epoch, optim, criterian):
        fig, ax = plt.subplots(3, 1)
        x = np.arange(epoch)
        y_loss = np.zeros_like(x)
        acc_train = np.zeros_like(x)
        acc_test = np.zeros_like(x)

        # 训练过程
        # for i in range(epoches):
#         y = m(X_train)
#         l = loss(y, Y_train.T) / 60000
#         l.backward()
#         SGD(m, 0.1)
#         zero_grad(m)
        for i in range(epoch):
#             optim.zero_grad()
            y_out = self.model(self.x_train)
            loss = criterian(y_out, self.y_train.T)/60000
            loss.backward()
            optim(self.model,alpha)
#             optim.step()
            
            y_loss[i] = loss.item

            # 计算训练数据准确率
            y_pred = y_out.argmax(axis=0)
            acc_train[i] = (y_pred == self.y_train).sum() / y_pred.shape

            # 计算测试数据准确率
            y_pred = self.model(self.x_test).argmax(axis=0)
            acc_test[i] = (y_pred == self.y_test).sum() / y_pred.shape

        ax[0].plot(x, y_loss)
        ax[0].set_title('Loss')

        ax[1].plot(x, acc_train)
        ax[1].set_title('Train acc')

        ax[2].plot(x, acc_test)
        ax[2].set_title('Test acc')

        fig.show()
