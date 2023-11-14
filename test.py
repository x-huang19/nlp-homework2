import torch
import torch.nn as nn
import jieba

sentence = '对于“一地两检”方案未来会否受到挑战，张达明认为成功几率很低。'
for subword in jieba.cut(sentence):
    print(subword)