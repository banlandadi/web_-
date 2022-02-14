# README

PB18111760 王嘉梁

PB18111782 徐瑞

### 一、运行环境

```python
from gensim.models import word2vec
from collections import defaultdict
import numpy as np
from torch import nn
import torch
```

python==3.8     windows10     intel-i5     

### 二、运行方式

直接使用python运行main.py即可

### 三、关键函数说明

`def get_w2v_emb(input_path, output_path):` 读取实体描述和关系描述，提取实体对应的特征向量

`def create_new_files():`创建新格式文件，测试和debug用

`def triple_get(path):`读取三元组

`def my_index(my_list, entitys, my_flag):`  构建实体名和index的字典，方便使用

`class KGE(nn.Module):`KGE模型，内含RESCAL和DisMult两种计算方式

`def train(Model, train_path, entitys, epochs, batch_size, optimizer, loss_func):`根据参数训练模型

`def test(Model, test_path, entitys, entitys_reverse, result_path, train_data):`读取测试数据，计算结果

`def main():`综合模块，定义各个文件的路径，完成功能



