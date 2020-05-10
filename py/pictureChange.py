# !/usr/bin/env python
# coding:utf-8
# Author: Lizechen
# 图片处理

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

np.set_printoptions(threshold=np.inf)

file = 'cifar-10-batches-py/data_batch_1'
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


imgdict = unpickle(file)
print(imgdict.keys())
print(imgdict[b'filenames'])
print(imgdict[b'labels'])
# 选一张图片
cat_data = imgdict[b'data'][7]
cat_data = cat_data.reshape(3, 32, 32)
# 得到RGB通道
r = Image.fromarray(cat_data[0]).convert('L')
g = Image.fromarray(cat_data[1]).convert('L')
b = Image.fromarray(cat_data[2]).convert('L')
image = Image.merge("RGB", (r, g, b))
# 显示图片
plt.imshow(image)
plt.show()

with open('truck_r.txt', 'w') as file:
    for num in cat_data[0].reshape(1024):
        file.write(str(num)+'\n')
with open('truck_g.txt', 'w') as file:
    for num in cat_data[1].reshape(1024):
        file.write(str(num) + '\n')
with open('truck_b.txt', 'w') as file:
    for num in cat_data[2].reshape(1024):
        file.write(str(num) + '\n')