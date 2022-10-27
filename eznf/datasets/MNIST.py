import requests
import time
import os
import numpy as np
import gzip
import random
import matplotlib.pyplot as plt
from io import BytesIO

class MNIST:
    url = 'http://yann.lecun.com/exdb/mnist/'
    resources = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
    
    def __init__(self, root:str, download:bool, shuffle:bool = False):
        self.root = root 
        self.download = download
        self.shuffle = shuffle
        
    def __unpack(self):
        '''Return
            data->list
        '''
        data = []
        for idx, val in enumerate(MNIST.resources):
            file_gz = MNIST.url + val + '.gz'
            time.sleep(0.5)
            response = requests.get(file_gz)
            
            if self.download == True:
                if not os.path.exists(self.root + '\\' + val + '.gz'):
                    with open(self.root + '\\' + val + '.gz', 'wb') as file:
                        file.write(response.content)
                    
            buffer = BytesIO(response.content)
            g_file = gzip.GzipFile(fileobj = buffer)
            
            if idx == 0 or idx == 2:
                sub_data = np.frombuffer(g_file.read(), dtype = np.uint8)[16:].reshape(-1, 28, 28)
                data.append(sub_data)
                     
            elif idx == 1 or idx == 3:
                sub_data = np.frombuffer(g_file.read(), dtype = np.uint8)[8:]
                data.append(sub_data)

        return data
    
    def get(self):
        '''Returns:
           X_train -> Tensor: train_data 
           Y_train -> Tensor: train_label
           X_test -> Tensor: test_data
           Y_test -> Tensor: test_label
        '''
        data = self.__unpack()
        self.X_train, self.Y_train, self.X_test, self.Y_test = data[0], data[1], data[2], data[3]
            
        if self.shuffle == True:
            self.X_shuffle_train = []
            self.Y_shuffle_train = []
            idxs = np.random.permutation(self.X_train.shape[0])
            for idx,val in enumerate(idxs):
                self.X_shuffle_train.append(self.X_train[val])
                self.Y_shuffle_train.append(self.Y_train[val])
                
            return np.array(self.X_shuffle_train), np.array(self.Y_shuffle_train), self.X_test, self.Y_test
                
        else:
            return self.X_train, self.Y_train, self.X_test, self.Y_test
        
    def show_one_sample(self):
        '''Return
            Null
        '''
        idx = random.randint(0, 10000)
        plt.imshow(self.X_test[idx].reshape(-1, 28), cmap = 'gray')
        print('label:{}'.format(self.Y_test[idx]))