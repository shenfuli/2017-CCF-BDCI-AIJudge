## mac 下 xgboost 的安装
https://www.jianshu.com/p/76ff402a8b58 <br> 
https://www.jianshu.com/p/c2b0c3067d84 <br> 
我们采用源码的方式编译并安装xgboost
## 1. 安装xgboost
第一步：克隆最新的XGBoost到本地

git clone --recursive https://github.com/dmlc/xgboost

可以通过build.sh 进行编译，支持多线程xgboost

第二步：安装gcc

brew install gcc5 --without-multilib


第三步：修改XGBoost的config文件

cd xgboost
cp make/config.mk ./config.mk

很多攻略都是直接将以下这两行
```
# export CC = gcc
# export CXX = g++

改为：
直接把config.mk修改成：
export CC = /usr/local/bin/gcc-5
export CXX = /usr/local/bin/g++-5
```


第四步：开始编译

make clean_all && make -j4


第五步：安装python包

cd python-package
python setup.py install

至些XGBoost终于安装成功！
在python环境中测试一下：
```
mymac:xgboost zhengwenjie$ ipython
Python 3.7.2 (default, Dec 29 2018, 00:00:04) 
Type 'copyright', 'credits' or 'license' for more information
IPython 6.5.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import xgboost as xgb

In [2]: xgb.__version__
Out[2]: '0.83.dev0'

``` 
执行完成后，没有出现错误提示，表明成功安装。

 