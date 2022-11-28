# eznf框架使用方法

### 包依赖

见 `./requirements.txt`



### 使用方法

`import eznf`



### 数据集

*使用MNIST的逻辑与pytorch相同 ； 特别说明，使用cifar10数据集，用户创建cifar10实例后，必须调用其download方法下载数据集gz，然后用户需要自己解压，并将解压后的文件夹里的路径传入实例的get方法，即可以返回数据*

```python
from eznf import datasets

mnist = datasets.MNIST('./', True)	#下载
data_m = mnist.get()

cifar10 = datasets.Cifar10('./', True)	#下载
data_c = cifar10.get()
```



### Tensor使用

具体操作同`pytorch`，支持GPU

```python
tensor = eznf.Tensor(2, 4)
print(tensor)

>>>tensor(
	[[ 1.30387739 -1.1998284   1.47868658  0.46624838]
	 [-0.56039362 -1.57864911 -0.9321185  -0.69342469]]
	)
```



### 模型构建

示例：

```python
class CNN(eznf.nn.Module):
    def __init__(self):
        super().__init__()
        self.networks = [
            eznf.nn.Cov2d(1, 3, 3),
            eznf.nn.MaxPooling(2),
            eznf.nn.Flatten(),
            eznf.nn.Linear(507, 256),
            eznf.nn.ReLU(),
            eznf.nn.Linear(256, 10)
        ]
    
    def forward(self, x):
        for i in self.networks:
            x = i(x)
        return x
```
