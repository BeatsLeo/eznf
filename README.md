# eznf框架

---

## 项目需求

### Tensor

`eznf/tensor/tensor.py`

* 支持GPU

* 支持list、ndarray的相互转换

* 重载运算符

* 实现backward



### Module

`eznf/nn/modules`

基类：Module

* 支持GPU

#### Linear

#### ReLu

#### Sigmoid

#### Tanh

#### Softmax

#### Cov1d

#### Cov2d

#### Cov3d

#### MaxPool

#### AvgPool

#### MSELoss

#### CrossEntropyLoss

#### Hebb

#### 感知机

(时间允许：BatchNorm, Dropout)



### Functional

`eznf/nn/functional.py`

基类：被使用于module

* 支持GPU

#### linear

#### reLu

#### sigmoid

#### tanh

#### softmax

#### cov1d

#### cov2d

#### cov3d

#### maxPool

#### avgPool

#### mse_loss

#### cross_entropy



### Optimizer

`eznf/optim`

* 支持GPU

#### SGD

#### Adam



### DataSet

`eznf/dataset`

#### MNIST

#### Cifar



### DataLoad

`eznf/dataload`

* 支持GPU

* 封装训练集和测试集

（支持batch）



### Visualization

`eznf/visualization`

#### Train

#### CM