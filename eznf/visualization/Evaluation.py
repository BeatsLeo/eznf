import matplotlib.pyplot as plt
import numpy as np
class Evaluation(object):
    def __init__(self, model, x_train, x_test, y_train, y_test, n):
        self.y_train = y_train.argmax(axis=1)
        self.y_test = y_test.argmax(axis=1)
        self.n = n
        self.y_train_out = model(x_train)
        self.y_test_out = model(x_test)
        self.train_priedict = self.y_train_out.argmax(axis=0)
        self.test_priedict = self.y_test_out.argmax(axis=0)
        
    def CMplot(self):
        C1 = np.zeros((self.n, self.n))
        C2 = np.zeros((self.n, self.n))

        for i in range(self.y_train.shape[0]):
            C1[self.y_train[i].item, self.train_priedict[i].item] += 1

        for i in range(self.y_test.shape[0]):
            C2[self.y_test[i].item, self.test_priedict[i].item] += 1

        img, ax = plt.subplots(1, 2)

        ax[0].matshow(C1, cmap=plt.cm.Blues)
        ax[0].set_title('Train')
        ax[0].set_xlabel('priedict')
        ax[0].set_ylabel('real')

        ax[1].matshow(C2, cmap=plt.cm.Reds)
        ax[1].set_title('Test')
        ax[1].set_xlabel('priedict')
        ax[1].set_ylabel('real')
        img.show()

    def ROCplot(self):
        L = 100
        fig, ax = plt.subplots()
        for i in range(self.n):
            # 第i类
            tpr = np.zeros(L)
            fpr = np.zeros(L)
            for j in range(L):
                r = j * (1 / L)  # 第j个阈值
                tp = 0
                fp = 0
                fn = 0
                tn = 0
                for k in range(self.y_test.shape[0]):
                    p = (self.y_test_out[i, k].item >= r)
                    t = (self.y_test.item[k] == i)
                    
                
                    if p == 1 and t == 1:
                        tp += 1
                    if p == 1 and t == 0:
                        fp += 1
                    if p == 0 and t == 1:
                        fn += 1
                    if p == 0 and t == 0:
                        tn += 1
                tpr[j] = tp / (tp + fn)
                fpr[j] = fp / (fp + tn)
            # 画出第i类的ROC曲线
            ax.plot(fpr, tpr, label='Class%i' % i)

        ax.set_title('ROC')
        ax.set_xlabel('fpr')
        ax.set_ylabel('tpr')
        ax.legend()
        fig.show()

    def PRplot(self):
        L = 100
        fig, ax = plt.subplots()

        for i in range(self.n):
            precision = np.zeros(L)
            recall = np.zeros(L)
            for j in range(L):
                r = j * (1 / L)  # 阈值
                tp = 0
                fp = 0
                fn = 0
                tn = 0
                for k in range(self.y_test.shape[0]):
                    p = (self.y_test_out[i, k].item >= r)
                    t = (self.y_test.item[k] == i)

                    if p == 1 and t == 1:
                        tp += 1
                    if p == 1 and t == 0:
                        fp += 1
                    if p == 0 and t == 1:
                        fn += 1
                    if p == 0 and t == 0:
                        tn += 1

                precision[j] = tp / (tp + fp)
                recall[j] = tp / (tp + fn)

            # 画出第i类的pr曲线
            ax.plot(recall, precision, label='Class%i' % i)

        ax.set_title('PR')
        ax.set_xlabel('fpr')
        ax.set_ylabel('tpr')
        ax.legend()
        fig.show()


