# mac install pytorch 


## 1. install pytorch
conda install pytorch torchvision -c pytorch

## 2. install torchvision
pip3 install https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-none-macosx_10_7_x86_64.whl

pip3 install torchvision

```
直接pip3 install https://xxxxx, 失败后手动下载文件，然后进行安装

下载 https://download.pytorch.org/whl/cpu/torch-1.0.0-cp37-none-macosx_10_7_x86_64.whl 后，然后安装
$ pip3 install torch-1.0.0-cp37-none-macosx_10_7_x86_64.whl 
Looking in indexes: http://mirrors.aliyun.com/pypi/simple/
Processing ./torch-1.0.0-cp37-none-macosx_10_7_x86_64.whl
Installing collected packages: torch
  Found existing installation: torch 1.0.1.post2
    Uninstalling torch-1.0.1.post2:
      Successfully uninstalled torch-1.0.1.post2
Successfully installed torch-1.0.0

```
## 3. verfication pytorch program 
ipython 

enter function pages： 

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

## reference
1. pytorch 安装步骤
https://pytorch.org/get-started/locally/
2. CPU 模式下安装pytorch
https://pytorch.org/get-started/previous-versions/