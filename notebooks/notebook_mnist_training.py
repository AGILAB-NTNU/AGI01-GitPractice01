#%%

import torch
from torch import nn
import torchvision as tv
from IPython.display import clear_output
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

train_data = tv.datasets.MNIST("../dataset/MNIST/", train=True, transform=None,target_transform=None,download=True) #下載並匯入MNIST訓練資料
test_data = tv.datasets.MNIST("../dataset/MNIST/", train=False, transform=None,target_transform=None,download=True) #下載並匯入MNIST測試資料

print('Number of samples in train_data is: ',len(train_data))
print('Number of samples in test_data is: ',len(test_data))
# %%

from matplotlib import pyplot as plt
import numpy as np

x = train_data.data[0]
plt.imshow(x)
# %%
