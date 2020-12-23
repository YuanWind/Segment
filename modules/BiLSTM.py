# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : BiLSTM.py
import torch
import torch.nn as nn
class BiLSTM(nn.Module):
    def __init__(self,vocab,config):
        super(BiLSTM,self).__init__()
        self.config=config
        self.vocab=vocab
        self.char_embed=nn.Embedding(vocab.char_size,config.char_dims,padding_idx=0)
        self.bichar_embed=nn.Embedding(vocab.bichar_size,config.bichar_dims,padding_idx=0)

        self.exchar_embed = nn.Embedding(vocab.exchar_size, config.char_dims, padding_idx=0)
        self.exbichar_embed = nn.Embedding(vocab.exbichar_size, config.bichar_dims, padding_idx=0)

        self.exchar_embed.weight.requires_grad = False
        self.exbichar_embed.weight.requires_grad = False

        self.char_liner=nn.Linear(config.char_dims*2+config.bichar_dims*2,config.input_dims)
        self.lstm = nn.LSTM(
            input_size=config.char_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,

        )
        self.liner=nn.Linear(in_features=config.lstm_hiddens * 2,
                               out_features=vocab.label_size,
                               bias=False)
        nn.init.kaiming_uniform_(self.liner.weight)
    def forward(self,batch_chars,batch_bichars,batch_exchars,batch_exbichars,char_mask):

        real_length=[torch.sum(char_mask[i]) for i in range(len(batch_chars))]
        chars_embed=self.char_embed(batch_chars)
        bichars_embed=self.bichar_embed(batch_bichars)
        ex_chars_embed=self.exchar_embed(batch_exchars)
        ex_bichars_embed=self.exbichar_embed(batch_exbichars)
        # if self.training:
        #     chars_emb = chars_embed
        char_represents = torch.cat([chars_embed,bichars_embed,ex_chars_embed,ex_bichars_embed], -1)
        lstm_input=torch.tanh(self.char_liner(char_represents))
        lstm_hidden, _ = self.lstm(lstm_input)
        score = self.liner(lstm_hidden)
        return score

    def init_pretrain_vec(self,exchar_embed,exbichar_embed):

        self.char_embed.weight.data.copy_(torch.zeros([self.vocab.char_size,self.config.char_dims]))
        self.bichar_embed.weight.data.copy_(torch.zeros([self.vocab.bichar_size,self.config.bichar_dims]))

        self.exchar_embed.weight.data.copy_(exchar_embed)
        self.exbichar_embed.weight.data.copy_(exbichar_embed)