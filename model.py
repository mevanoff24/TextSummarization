import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
import numpy as np
import random 
import sys

# we keep the word embedding not trainable, you can adjust if desired. 

def create_embeddings(vect, idx2word, emb_size, pad_idx=1):
    emb = nn.Embedding(len(idx2word), emb_size, padding_idx=pad_idx)
    weights = emb.weight.data
    for i, word in enumerate(idx2word):
        try:
            weights[i] = torch.from_numpy(vect[word] * 3)
        except:
            pass

    emb.weight.requires_grad = False
    return emb

def rand_t(*sz): 
    return torch.randn(sz) / np.sqrt(sz[0])

def rand_p(*sz): 
    return nn.Parameter(rand_t(*sz))



class Seq2SeqAttention(nn.Module):
    def __init__(self, vecs_enc, idx2word_enc, em_sz_enc, vecs_dec, idx2word_dec, em_sz_dec, 
                 num_hidden, out_seq_length, num_layers=2, activation=F.tanh, pad_idx=1):
        super().__init__()
        self.num_hidden = num_hidden
        self.out_seq_length = out_seq_length
        self.num_layers = num_layers
        self.activation = activation
        # encoder
        self.encoder_embeddings = create_embeddings(vecs_enc, idx2word_enc, em_sz_enc, pad_idx)
        self.encoder_dropout_emb = nn.Dropout(0.1)
        self.encoder_dropout = nn.Dropout(0.1)
        self.encoder_gru = nn.GRU(em_sz_enc, num_hidden, num_layers=num_layers, bidirectional=True)
        self.encoder_out = nn.Linear(num_hidden*2, em_sz_dec, bias=False)
        # decoder
        self.decoder_embeddings = create_embeddings(vecs_dec, idx2word_dec, em_sz_dec, pad_idx)
        self.decoder_dropout = nn.Dropout(0.1)
        self.decoder_gru = nn.GRU(em_sz_dec, em_sz_dec, num_layers=num_layers)
        self.out = nn.Linear(num_hidden, len(idx2word_dec))
        self.out.weight.data = self.decoder_embeddings.weight.data
        # attention
        self.W1 = rand_p(num_hidden*2, em_sz_dec)
        self.l2 = nn.Linear(em_sz_dec, em_sz_dec)
        self.l3 = nn.Linear(em_sz_dec+num_hidden*2, em_sz_dec)
        self.V = rand_p(em_sz_dec)        

        
    def forward(self, X, y=None, tf_ratio=0.0, return_attention=False):
        # encode forward
        seq_len, batch_size = X.size()
        hidden = self.initHidden(batch_size)
        enc_embs = self.encoder_dropout_emb(self.encoder_embeddings(X))
        enc_out, hidden = self.encoder_gru(enc_embs, hidden)
        hidden = hidden.view(2, 2, batch_size, -1).permute(0, 2, 1, 3).contiguous().view(2, batch_size, -1)
        hidden = self.encoder_out(self.encoder_dropout(hidden))
        # decode forward
        dec_input = Variable(torch.zeros(batch_size).long()).cuda()
        w1e = enc_out @ self.W1
        results = []
        attentions = []
        for i in range(self.out_seq_length):
            w2d = self.l2(hidden[-1])
            u = self.activation(w1e + w2d)
            a = F.softmax(u @ self.V, dim=0)
            attentions.append(a)
            Xa = (a.unsqueeze(2) * enc_out).sum(0)
            dec_embs = self.decoder_embeddings(dec_input)
            weight_enc = self.l3(torch.cat([dec_embs, Xa], dim=1))
            outp, hidden = self.decoder_gru(weight_enc.unsqueeze(0), hidden)
            outp = self.out(self.decoder_dropout(outp[0]))
            results.append(outp)
            # teacher forcing
            dec_input = Variable(outp.data.max(1)[1]).cuda()
            if (dec_input == 1).all():
                break
            if (y is not None) and (random.random() < tf_ratio):
                if i >= len(y): 
                    break
                # assign next value to decoder input
                dec_input = y[i]
        if return_attention:
            return torch.stack(results), torch.stack(attentions)
        else:
            return torch.stack(results)
        
        
    def initHidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers*2, batch_size, self.num_hidden)).cuda()
        