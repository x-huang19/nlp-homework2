import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
import time
import math
import re
import jieba
from torch.utils.data import DataLoader, Dataset
#Models Class
class FNNLM(nn.Module):

    def __init__(self, vocab_size, embedded_size, context_size):
        super(FNNLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedded_size) # 生成embedding矩阵
        self.linear1 = nn.Linear(context_size * embedded_size, embedded_size)
        self.linear2 = nn.Linear(embedded_size, vocab_size)
        #Drop out function
        self.dropout = nn.Dropout(p)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1)) # 由输入index获取每个词对应的低纬向量，并使用view方法拼接
        out = F.relu(self.linear1(embeds))
        out = self.dropout(out)
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        
        
class dataset(Dataset):
    # 根据分词后训练集，每三个词之间为3gram
    def __init__(self, trigram, voc, index):
        
        self.context = []
        self.labels = []
        for data in trigram:
            temp = []
            for word in data[0]:
                if word in voc.keys():
                    temp.append(voc[word])
                else:
                    temp.append('unk')
            if data[1] in voc.keys():
                target = voc[data[1]]
            else:
                target = voc['unk']
            self.context.append(torch.tensor(temp, dtype=torch.long))   
            self.labels.append(torch.tensor(target, dtype=torch.long))  
            
        self.nsamples = len(trigram)
    
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, index):
        
        context = self.context[index]
        label = self.labels[index]
        return context, label
    
def dataloader(path, batch_size):
    
    mydataset = dataset(path)
    return DataLoader(mydataset, batch_siz=batch_size, shuffle=False)

if __name__ == "__main__":
    
    # 超参数定义
    p = 0.5
    index = 0
    vocab_size = 0
    embedded_size = 100
    context_size = 3
    th = 200
    vocab_dict = {} # 统计每个词出现频率
    tokens = {} # 根据阈值筛选词
    index_dict = {} #
    
    # part1 数据预处理 根据语料库获取词典
    filepath = './dataset/news.2017.zh.shuffled.deduped'
    f = open(filepath,'r',encoding='utf-8')
    context = f.readlines()
    totalnum = len(context)
    trainsize = totalnum - 1000
    trigram = [] # ([word1, word2, word3], label)
    
    for m in range(trainsize):
        # 首先应该对数据进行清洗
        sentence = re.sub(r'[^\u4e00-\u9fa5，。]', '', context[m])
        cutwords = jieba.lcut(sentence)
        for subword in cutwords:
            if subword not in vocab_dict.keys():
                vocab_dict[subword] = 1
            else:
                vocab_dict[subword] += 1

    for word in vocab_dict.keys():
        if vocab_dict[word] > th:
            tokens[word] = index
            index_dict[index] = word
            index += 1
    
    tokens['unk'] = index
    index_dict[index] = 'unk'
    vocab_size = index + 1
    
    # part2 根据语料库，划分数据集trigram
    for m in range(trainsize):
    
        sentence = re.sub(r'[^\u4e00-\u9fa5，。]', '', context[m])
        cutwords = jieba.lcut(sentence)
        words = []
        
        for n in range(len(cutwords) - 2):
            print('s')
        
    f.close()  

    