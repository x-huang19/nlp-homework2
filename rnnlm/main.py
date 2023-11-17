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

class RNNLanguageModler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNLanguageModler, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=1)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs)
        output, _ = self.rnn(embeds)
        out = self.linear(output)
        out = F.log_softmax(out, dim = 2)
        return out

def process(processed):
    
    index = 0
    th = 200
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
        corpus_train = []
        corpus_eval = []
        f.close()
        
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
        
        # 获取rnn中所取训练集 context, label (长度为L)
        for m in range(trainsize):
            
            words = []
            sentence = re.sub(r'[^\u4e00-\u9fa5，。]', '', context[m])
            cutwords = jieba.lcut(sentence)
            
            for cell in cutwords:
                if cell in tokens.keys():
                    words.append(cell)
                else:
                    words.append('unk')
            text = words[:-1]
            label = words[1:]
            corpus_train.append((text, label))
            
        for m in range(trainsize, totalnum):
            
            sentence = re.sub(r'[^\u4e00-\u9fa5，。]', '', context[m])
            cutwords = jieba.lcut(sentence)
            words = []
            for cell in cutwords:
                if cell in tokens.keys():
                    words.append(cell)
                else:
                    words.append('unk')
            text = words[:-1]
            label = words[1:]
            corpus_eval.append((text, label))
            
        with open('./corpus_train.pickle','wb') as datafile:
            pickle.dump(corpus_train, datafile)
            
        with open('./corpus_eval.pickle','wb') as datafile:
            pickle.dump(corpus_eval, datafile)
        
        with open('./tokens.pickle','wb') as datafile:
            pickle.dump(tokens, datafile) 
            
    else:
        
        with open('./corpus_train.pickle','rb') as datafile:
            corpus_train = pickle.load(datafile)
            
        with open('./corpus_eval.pickle','rb') as datafile:
            corpus_eval = pickle.load(datafile)
        
        with open('./tokens.pickle','rb') as datafile:
            tokens = pickle.load(datafile)
                                 
    return tokens, corpus_eval, corpus_train

class dataset(Dataset):
    # 根据分词后训练集，每三个词之间为3gram
    def __init__(self, corpus, voc):
        
        self.context = []
        self.labels = []

        for data in corpus:
            
            temp = []
            for word in data[0]:
                if word in voc.keys():
                    temp.append(voc[word])
                else:
                    temp.append('unk')
            self.context.append(torch.tensor(temp, dtype=torch.long)) 
            
            labels = []
            for word in data[1]:
                if word in voc.keys():
                    labels.append(voc[word])
                else:
                    labels.append('unk')               
            self.labels.append(torch.tensor(labels, dtype=torch.long))  
            
        self.nsamples = len(corpus)
    
    def __len__(self):
        return self.nsamples
    
    def __getitem__(self, index):
        
        context = self.context[index]
        label = self.labels[index]
        return context, label
    
def dataloader(corpus, voc, batch_size):
    
    mydataset = dataset(corpus, voc)
    return DataLoader(mydataset, batch_size=batch_size, shuffle=False)

def train(model, dataloader_train, loss_function, optimizer, device, batch_size, epoch, dataloader_valid, scheduler):
    
    total_accu = None
    total_loss = 0
    file1 = open("log.txt", "a")
    file1.write('start training\n')
    losses = []
    BATCH_NUM = 1000
    for epochnum in range(1, epoch + 1):
        
        model.train()
        start = time.time()
        totalcount = 0
        counter = 0
        for index, text in enumerate(dataloader_train):
            
            label = text[1].to(device)
            text = text[0].to(device)
            L = len(text[0])
            if L == 0:
                continue
            optimizer.zero_grad()
            log_probs = model(text) # shape N L C
            loss = None
            for l in range(L):
                if loss == None:
                    loss = loss_function(log_probs[0, l, :], label[0, l])
                else:
                    loss += loss_function(log_probs[0, l, :], label[0, l])       
            loss = loss / L
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
                
                label = text[1].to(device)
                text = text[0].to(device)
                L = len(text[0])
                if L == 0:
                    continue
                log_probs = model(text).to(device)
                loss = None
                for l in range(L):
                    if loss == None:
                        loss = loss_function(log_probs[0, l, :], label[0, l])
                    else:
                        loss += loss_function(log_probs[0, l, :], label[0, l])  
                        
                loss = loss / L                   
                total_loss_eval += loss.item()
                total_acc += (log_probs[0].argmax(1) == label[0]).sum().item()
                total_count += L
        
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
    
def model_test(vocab_size, hidden_size , embedded_size):
    
    model = RNNLanguageModler(vocab_size, embedded_size, hidden_size)
    s = [3, 8, 91, 129]
    s = torch.tensor(s, dtype=torch.long).unsqueeze(dim=0)
    out = model(s)
    print(out.shape)

if __name__ == "__main__":
    
    torch.manual_seed(1213)
    lr = 0.05
    batch_size = 1
    p = 0.5
    epochs = 2
    index = 0
    embedded_size = 16
    hidden_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens, corpus_eval, corpus_train = process(True)
    vocab_size = len(tokens.keys())
    model = RNNLanguageModler(vocab_size, embedded_size, hidden_size)
    # model_test(vocab_size = 200, hidden_size = 10, embedded_size = 16)
    mydataloader = dataloader(corpus_train, tokens, batch_size)
    eval_dataloader = dataloader(corpus_eval, tokens, batch_size)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    #Changes Learning Rate according optimization
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1) 
    train(model, mydataloader, loss_function, optimizer, device, batch_size, epochs, eval_dataloader, scheduler)   
     
