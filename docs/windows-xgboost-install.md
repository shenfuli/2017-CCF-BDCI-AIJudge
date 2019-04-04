## windows 下py-xgboost 的安装
https://www.zhihu.com/question/46377605 <br> 
https://anaconda.org/anaconda/py-xgboost

## 1. 安装py-xgboost
conda install -c anaconda py-xgboost  <br> 
备注： 这里已经安装好conda,这里不在介绍
## 2. 下载 xgboost-0.82-cp36-cp36m-win_amd64.whl并安装
pip install  xgboost-0.82-cp36-cp36m-win_amd64.whl --user <br>
备注： --user 表示当前用户下安装
## 3. python 测试xgboost 是否成功安装
windows 下运行ipython，然后进入ipython的工作界面，执行下面命令
```
In [1]: import xgboost as xgb 

In [2]:
``` 
执行完成后，没有出现错误提示，表明成功安装。

 