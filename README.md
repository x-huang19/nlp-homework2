# nlp-homework2
saving code for course nlp

## homework description
使用前端神经网络、循环神经网络以及自注意力网络构建语言模型，并得到相应困惑度

## 构建流程

### 分词
训练集每行为一个句子，使用分词工具对每一行分词，构建训练集词典

### 前端神经网络

在模型初始化阶段由nn.Embedding构建词向量tableLt，在分词构建词典过程中每个词以下标获取tabelLt上的对应向量

训练方法：

网络输入数据是index，如二元LM，输入context(index1,index2)，label(index3)