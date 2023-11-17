import torch
import torch.nn as nn
import jieba


a = torch.randn((1,3,10))
print(a)
print(a[0].argmax(1))
