import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
class VTrain(object):
    def __init__(self, model, x_train, x_test, y_train, y_test):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        

    def train(self, alpha, epoch, optim, criterian,batch_size):
        fig, ax = plt.subplots()
        px = np.arange(epoch)
        y_loss = np.zeros_like(px)
        acc_train = np.zeros_like(px)
        acc_test = np.zeros_like(px)

        # 训练过程
        with tqdm(total=epoch) as t:
            for i in range(epoch):
                for j in range(len(self.x_train) // batch_size):
                    x = self.x_train[j*batch_size : (j+1)*batch_size]
                    y = self.y_train[j*batch_size : (j+1)*batch_size]
                    out = self.model(x.T)
                    l =  criterian(out, y.T) / batch_size
                    l.backward()
                    optim(self.model, 0.01)
                    for w in self.model.parameters():
                        w.grad = None
                    
                y_loss[i]=l.item[0]
                
                y_pred = self.model(self.x_train.T).argmax(axis=0).item

                acc_train[i] = (y_pred == self.y_train.argmax(axis=1).item).sum() / y_pred.shape
                
                y_pred = self.model(self.x_test.T).argmax(axis=0).item
                acc_test[i] = (y_pred == self.y_test.argmax(axis=1).item).sum() / y_pred.shape                
                
                t.set_description('Epoch {}'.format(i), refresh=False)
                t.set_postfix(loss=l.item[0], refresh=False)
                t.update(1)

        ax.plot(px, y_loss)
        ax.set_title('Loss')

        ax[1].plot(px, acc_train)
        ax[1].set_title('Train acc')

        ax[2].plot(px, acc_test)
        ax[2].set_title('Test acc')

        fig.show()
