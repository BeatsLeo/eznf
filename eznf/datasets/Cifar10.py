import requests
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

class Cifar10:
    url = 'http://www.cs.toronto.edu/~kriz'
    resource = ['cifar-10-python.tar.gz']
    resources = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

    def __init__(self, root:str):
        self.root = root

    def __unpickle(self, path:str):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []

        for _, val in enumerate(Cifar10.resources):
            if val != 'test_batch':
                file = path + '\\' + val
                with open(file, 'rb') as f:
                    dict = pickle.load(f, encoding = 'bytes')
        
                    for _, val in enumerate(dict[b'data']):
                        X_train.append(val)
        
                    for val in dict[b'labels']:
                        Y_train.append(val)
            else:
                file = path + '\\' + val
                with open(file, 'rb') as f:
                    dict = pickle.load(f, encoding = 'bytes')
        
                    for _, val in enumerate(dict[b'data']):
                        X_test.append(val)
        
                    for val in dict[b'labels']:
                        Y_test.append(val)
        return X_train, Y_train, X_test, Y_test

    def download(self):
        '''
        Return:
        path -> str
        '''
        file_gz = Cifar10.url + '/' + Cifar10.resource[0]
        time.sleep(0.5)
        response = requests.get(file_gz)

        if not os.path.exists(self.root + '\\' + Cifar10.resource[0]):
            with open(self.root + '\\' + Cifar10.resource[0], 'wb') as file:
                file.write(response.content)


        return self.root + '\\' + 'cifar-10-batches-py'
            
    def get(self, path:str):
        X_train, Y_train, X_test, Y_test = self.__unpickle(path)
        return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)