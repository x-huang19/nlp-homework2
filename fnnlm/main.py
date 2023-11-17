import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import string
import time
import math
import re
import jieba
import pickle
from torch.utils.data import DataLoader, Dataset
#Models Class
class FNNLM(nn.Module):

    def __init__(self, vocab_size, embedded_size, context_size, batch_size):
        super(FNNLM, self).__init__()
        
        self.batch_size = batch_size
        self.embeddings = nn.Embedding(vocab_size, embedded_size) # 生成embedding矩阵
        self.linear1 = nn.Linear(context_size * embedded_size, embedded_size)
        self.linear2 = nn.Linear(embedded_size, vocab_size)
        #Drop out function
        self.dropout = nn.Dropout(p)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((self.batch_size, -1)) # 由输入index获取每个词对应的低纬向量，并使用view方法拼接
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
    def __init__(self, trigram, voc):
        
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
    
def dataloader(trigram, voc, batch_size):
    
    mydataset = dataset(trigram, voc)
    return DataLoader(mydataset, batch_size=batch_size, shuffle=False)

def train(model, dataloader_train, loss_function, optimizer, device, batch_size, epoch, dataloader_valid, scheduler):
    
    # part1 training
    total_accu = None
    total_loss = 0
    file1 = open("log.txt", "a")
    file1.write('start training\n')
    losses = []
    BATCH_NUM = 10000
    
    for epochnum in range(1, epoch + 1):
        
        model.train()
        start = time.time()
        counter = 0
        totalcount = 0
        
        for index, text in enumerate(dataloader_train):
            
            if index == len(dataloader_train) - 1:
                continue
            label = text[1].to(device)
            text = text[0].to(device)
            optimizer.zero_grad()
            log_probs = model(text)
            loss = loss_function(log_probs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            losses.append(total_loss)
            counter += 1
            #Prints out log file every BATCH_NUM size
            if counter == BATCH_NUM:
                end = time.time()
                time_taken = end - start
                totalcount +=counter
                cur_loss = total_loss/totalcount
                #Write to file
                file1.write("\n")
                file1.write('| epoch {:3d} | {:5d}/{:5d} batches | ' 'lr {:02.2f} | ms/batches {:5.2f} | ' 'loss {:5.2f} | ppl {:8.2f}'.format(epochnum, totalcount, len(dataloader_train) , lr,time_taken,cur_loss, math.exp(cur_loss)))
                #System Write
                print('| epoch {:3d} | {:5d}/{:5d} batches | ' 'lr {:02.2f} | ms/batches {:5.2f} | ' 'loss {:5.2f} | ppl {:8.2f}'.format(epochnum, totalcount, len(dataloader_train) , lr,time_taken,cur_loss, math.exp(cur_loss)))
                counter = 0
                start = time.time()   
    # part2 eval
    #Evaluating model with validation data
        model.eval()
        total_acc, total_count = 0, 0
        total_loss_eval = 0
        with torch.no_grad():
            #Going through validation data batches
            for index, text in enumerate(dataloader_valid):
                
                if index == len(dataloader_valid) - 1:
                    continue
                
                label = text[1].to(device)
                text = text[0].to(device)
                predited_label = model(text).to(device)
                loss = loss_function(predited_label, label)
                total_loss_eval += loss.item()
                total_acc += (predited_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        #Creating logs from results
        time_taken = time.time() - start        
        accu_val = total_acc/total_count
        total_loss_eval = total_loss_eval/total_count
        ppl = math.exp(total_loss_eval)
        print("End of Epoch "+str(epochnum))
        print("Evaluation for validation data : "+str(accu_val))
        print("Time : " +str(time_taken))
        print("loss : "+str(total_loss_eval))
        print("ppl : "+str(ppl))
        file1.write("\nEnd of Epoch "+str(epochnum))
        file1.write("\nEvaluation for validation data : "+str(accu_val))
        file1.write("\nTime : " +str(time_taken))
        file1.write("\nloss : "+str(total_loss_eval))
        file1.write("\nppl : "+str(ppl))
        #Changing Learning rate
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        total_loss = 0    
        #Use path for Google collab "/content/gdrive/My Drive/model"
        torch.save(model, "model_{}".format(epochnum))
    file1.close()    

    
if __name__ == "__main__":
    
    # 超参数定义
    lr = 0.05
    batch_size = 20
    p = 0.5
    epochs = 2
    index = 0
    vocab_size = 0
    embedded_size = 128
    context_size = 3
    th = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed = 1
    
    if not processed:
        
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
        eval_trigram = [] # 测试集数据
        
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
        
        # part2 根据语料库，划分数据集trigram training
        for m in range(trainsize):
        
            sentence = re.sub(r'[^\u4e00-\u9fa5，。]', '', context[m])
            cutwords = jieba.lcut(sentence)
            words = []
            for cell in cutwords:
                if cell in tokens.keys():
                    words.append(cell)
                else:
                    words.append('unk')
            for n in range(len(words) - 3):
                trigram.append(([words[n],words[n + 1],words[n + 2]],words[n + 3]))
        
        for m in range(trainsize, totalnum):
            sentence = re.sub(r'[^\u4e00-\u9fa5，。]', '', context[m])
            cutwords = jieba.lcut(sentence)
            words = []
            for cell in cutwords:
                if cell in tokens.keys():
                    words.append(cell)
                else:
                    words.append('unk')
            for n in range(len(words) - 3):
                eval_trigram.append(([words[n],words[n + 1],words[n + 2]],words[n + 3]))        
        
        with open('./trigram.pickle','wb') as datafile:
            pickle.dump(trigram, datafile)
            
        with open('./eval_trigram.pickle','wb') as datafile:
            pickle.dump(eval_trigram, datafile)
        
        with open('./tokens.pickle','wb') as datafile:
            pickle.dump(tokens, datafile)    
    else:
        
        with open('./trigram.pickle','rb') as datafile:
            trigram = pickle.load(datafile)
            
        with open('./eval_trigram.pickle','rb') as datafile:
            eval_trigram = pickle.load(datafile)
        
        with open('./tokens.pickle','rb') as datafile:
            tokens = pickle.load(datafile)  
        
        vocab_size = len(tokens.keys())        
    
    # part3 training   
    model = FNNLM(vocab_size, embedded_size, context_size, batch_size).to(device)       
    mydataloader = dataloader(trigram, tokens, batch_size)
    eval_dataloader = dataloader(eval_trigram, tokens, batch_size)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #Changes Learning Rate according optimization
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1) 
    train(model, mydataloader, loss_function, optimizer, device, batch_size, epochs, eval_dataloader, scheduler)   
     
    f.close()  

    