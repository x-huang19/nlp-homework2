import torch
import torch.nn as nn
import jieba

n = 4
upper_tri = torch.tensor(float('-inf')).repeat((n, n))
lower_tri = torch.tril(torch.ones((n, n)))
matrix = upper_tri.triu(diagonal=1) + lower_tri

print(matrix)