# windows install pytorch 


## 1. install pytorch
conda install pytorch-cpu torchvision-cpu -c pytorch

## 2. install torchvision
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.1-cp36-cp36m-win_amd64.whl
pip3 install torchvision

note: this use python3, torchvision with python version matched 

## 3. verfication pytorch program 
ipython enter function pages： 

===> 
```
In [6]: from __future__ import print_function

In [7]: import torch

In [8]: x = torch.rand(5,3)

In [9]: print(x)
tensor([[0.9826, 0.0903, 0.1729],
        [0.8138, 0.7652, 0.4320],
        [0.5192, 0.7816, 0.6091],
        [0.9613, 0.1501, 0.9096],
        [0.6060, 0.6852, 0.4438]])
```

按照官方网站按照就没有什么问题，大功告成，类似于mac/linux/ubuntu 下的按照方案类似，不在详述
## reference
1. pytorch 安装步骤
https://pytorch.org/get-started/locally/
2. CPU 模式下安装pytorch
https://pytorch.org/get-started/previous-versions/