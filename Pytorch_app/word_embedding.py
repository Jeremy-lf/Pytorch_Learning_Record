import torch
from torch import nn
from torch.autograd import Variable


word_to_ix = {'hello':0,'word':1}
embeds = nn.Embedding(2,5)#nn.Embedding(m,n) m单词数目，n词嵌入的维度
hello_idx = torch.LongTensor([word_to_ix['hello']])
hello_idx = Variable(hello_idx)
hello_embed = embeds(hello_idx)
print(hello_embed)

#####################################################################################
#N-Gram

context_size =2
embedding_size=10

test_sentence = 'I like English,it is a really interesting language. there is almost 300 millon ' \
                'people speak it'.split()

trigram = [((test_sentence[i],test_sentence[i+1]),test_sentence[i+2]) for i in range(len(test_sentence)-2)]

#encoding
vocb = set(test_sentence)
word_to_ix = {word:i for i,word in enumerate(vocb)}
idx_to_word = {word_to_ix[word]:word for word in word_to_ix}

class NGramModel(nn.Module):
    def __init__(self,vocb_size,context_size,n_dim):
        super(NGramModel,self).__init__()
        # self.n_word = vocb_size
        self.embedding = nn.Embedding(vocb_size,n_dim)
        self.linear1 = nn.Linear(context_size*n_dim,128)
        self.linear2 = nn.Linear(128,vocb_size)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1,-1)
        out = self.linear1(emb)
        out = self.relu(out)
        out = self.linear2(out)
        log_prob = torch.log_softmax(out)

        return log_prob