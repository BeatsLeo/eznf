import requests
import time
import os
import numpy as np
import struct
import gzip
import random
import matplotlib.pyplot as plt

class MNIST:
    url = 'http://yann.lecun.com/exdb/mnist/'
    resources = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']
    
    def __init__(self, root:str, shuffle:bool, download:bool):
        self.root = root 
        self.download = download
        self.shuffle = shuffle
    
    def get(self):
        if self.download == True:
            data = []
            
            for idx, val in enumerate(MNIST.resources):    
                file_gz = MNIST.url + val + '.gz'
                time.sleep(0.5)
                response = requests.get(file_gz)
                
                filename = root + '\\' + val + '.gz'
                
                with open(filename, 'wb') as file:
                    file.write(response.content)
                
                g_file = gzip.GzipFile(filename)
                f_name = filename.replace(".gz", "")
                open(f_name, 'wb').write(g_file.read())
                g_file.close()
                
                with open(f_name, 'rb') as file:
                    if idx == 0 or idx == 2:
                        struct.unpack('>4i', file.read(16))
                        sub_data = np.fromfile(file, dtype = np.uint8).reshape(-1, 784)
                        data.append(sub_data)
                     
                    elif idx == 1 or idx == 3:
                        struct.unpack('>2i', file.read(8))
                        sub_data = np.fromfile(file, dtype = np.uint8)
                        data.append(sub_data)
                        
                os.remove(f_name)
            
            
            X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
            
            if self.shuffle == True:
                X_shuffle_train = []
                Y_shuffle_train = []
                idxs = np.random.permutation(X_train.shape[0])
                for idx,val in eunmerate(idxs):
                    X_shuffle_train[idx] = X_train[val]
                    Y_shuffle_train[idx] = Y_train[val]
                    
                return X_shuffle_train, Y_shuffle_train, X_test, Y_test
                
            else:
                return X_train, Y_train, X_test, Y_test
           
            
        
        else:
            data = []

            for idx, val in enumerate(MNIST.resources):
                file_gz = MNIST.url + val + '.gz'
                time.sleep(0.5)
                response = requests.get(file_gz)
                
                filename = root + '\\' + val + '.gz'
                with open(filename, 'wb') as file:
                    file.write(response.content)
                
                g_file = gzip.GzipFile(filename)
                
                f_name = filename.replace(".gz", "")
                open(f_name, 'wb').write(g_file.read())
                g_file.close()
                
                with open(f_name, 'rb') as file:
                    if idx == 0 or idx == 2:
                        struct.unpack('>4i', file.read(16))
                        sub_data = np.fromfile(file, dtype = np.uint8).reshape(-1, 784)
                        data.append(sub_data)
                     
                    elif idx == 1 or idx == 3:
                        struct.unpack('>2i', file.read(8))
                        sub_data = np.fromfile(file, dtype = np.uint8)
                        data.append(sub_data)
                os.remove(f_name)
                os.remove(filename)
            
            X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
            
            if self.shuffle == True:
                X_shuffle_train = []
                Y_shuffle_train = []
                idxs = np.random.permutation(X_train.shape[0])
                
                for idx,val in eunmerate(idxs):
                    X_shuffle_train[idx] = X_train[val]
                    Y_shuffle_train[idx] = Y_train[val]
                    
                return X_shuffle_train, Y_shuffle_train, X_test, Y_test
                
                
            else:
                return X_train, Y_train, X_test, Y_test
            
    def show_one_sample(self):
        idx = random.randint(0,60000)
        plt.imshow(X_train[idx].reshape(-1, 28), cmap = 'gray')
        print('label:{}'.format(Y_train[idx]))

#root = r'C:\Users\Lenovo\Desktop\python\dataset'
#test = MNIST(root = root, shuffle = False, download = False)
#X_train, Y_train, X_test, Y_test = test.get()