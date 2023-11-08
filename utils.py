import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
import time
import math
from torch.utils.data import DataLoader, Dataset
#Models Class
p = 0.9
vocab_size = 100
embedded_size = 30
context_size = 3
class FNNLM(nn.Module):

    def __init__(self, vocab_size, embedded_size, context_size):
        super(FNNLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedded_size)
        self.linear1 = nn.Linear(context_size * embedded_size, embedded_size)
        self.linear2 = nn.Linear(embedded_size, vocab_size)
        #Drop out function
        self.dropout = nn.Dropout(p)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
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
    # 根据分词后训练集，每两个词之间为2gram
    def __init__(self, path):
        
        self.context = []
        self.labels = []
        self.nsamples = 0
    
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, index):
        
        context = self.context[index]
        label = self.labels[index]
        return context, label
    
def dataloader(path, batch_size):
    
    mydataset = dataset(path)
    return DataLoader(mydataset, batch_siz=batch_size, shuffle=False)