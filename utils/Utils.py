# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : utils.py
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from utils import GobalVar
log_=GobalVar.get_value('log_')

def set_seed():
    random.seed(88)
    np.random.seed(88)
    torch.cuda.manual_seed(88)
    torch.manual_seed(88)
def get_GPU():
    gpu = torch.cuda.is_available()
    log1=("GPU available: ", gpu)
    log2=(" CuDNN: ", torch.backends.cudnn.enabled)
    print(log1,log2)
    log_.fprint_log(log1+log2)
    return gpu
def dump_pkl_data(data, f_name):
    with open(f_name, 'wb') as f:
        pickle.dump(data, f)
def load_pkl_data(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f)

def check_exist(dir,file):
    if not os.path.exists(dir):
        os.mkdir(dir)
    return os.path.join(dir,file)
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass
def reverse(x):
        rs={}
        for key,value in x.items():
            rs[value]=key
        return rs
def load_pretrained_vec(embfile,log_):
    embedding_dim = -1
    word_count = 0
    with open(embfile, encoding='utf-8') as f:
        for line in f.readlines():
            if word_count < 1:
                values = line.split()
                embedding_dim = len(values) - 1
            word_count += 1
    id2elem={0:'<padw>',1:'<unkw>'}
    index=len(id2elem)
    embeddings=torch.zeros([word_count + index,embedding_dim])
    with open(embfile, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            id2elem[len(id2elem)]=values[0]
            vector = torch.tensor(values[1:])
            embeddings[1] += vector
            embeddings[index] = vector
            index += 1
    embeddings[1] = embeddings[1] / word_count
    embeddings = embeddings / np.std(embeddings)
    elem2id = reverse(id2elem)
    if len(elem2id) != len(id2elem):
        log=("serious bug: extern words dumplicated, please check!")
        print(log)
        log_.fprint_log(log)
    return embeddings, id2elem, elem2id
def drop_input_independent(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
    scale = 3.0 / (2.0 * word_masks + 1e-12)
    word_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    return word_embeddings
def split_pretrained_vec(embfile):
    """在预训练文件中提取出char和bichar的向量，用pickle保存"""
    embedding_dim = -1
    char_count = 0
    bichar_count = 0
    with open(embfile, encoding='utf-8') as f:
        for line in f.readlines():
            values = line.split()
            if char_count < 1:
                embedding_dim = len(values) - 1
            if len(values[0])==1:
                char_count += 1
            if len(values[0])==2:
                bichar_count += 1
    char_id2elem={0:'<padc>',1:'<unkc>'}
    bichar_id2elem={0:'<padbc>',1:'<unkbc>'}
    char_index=len(char_id2elem)
    bichar_index=len(bichar_id2elem)
    char_embeddings=torch.zeros([char_count + char_index,embedding_dim])
    bichar_embeddings=torch.zeros([bichar_count + bichar_index,embedding_dim])
    with open(embfile, encoding='utf-8') as f:
        i=0
        for line in tqdm(f):
            if i==0:
                i+=1
                continue
            values = line.split()
            vector=[float(i) for i in values[1:]]
            vector = torch.tensor(vector)
            if len(values[0])==1:
                char_id2elem[len(char_id2elem)]=values[0]
                char_embeddings[1] += vector
                char_embeddings[char_index] = vector
                char_index += 1
            if len(values[0])==2:
                bichar_id2elem[len(bichar_id2elem)]=values[0]
                bichar_embeddings[1] += vector
                bichar_embeddings[bichar_index] = vector
                bichar_index += 1
    char_embeddings[1] = char_embeddings[1] / char_count
    bichar_embeddings[1] = bichar_embeddings[1] / bichar_count
    #char_embeddings = char_embeddings / torch.std(char_embeddings)
    #bichar_embeddings = bichar_embeddings / torch.std(bichar_embeddings)
    char_elem2id = reverse(char_id2elem)
    bichar_elem2id = reverse(bichar_id2elem)
    if len(char_elem2id) != len(char_id2elem):
        log=("serious bug: extern words dumplicated, please check!")
        print(log)
        log_.fprint_log(log)
    if len(bichar_elem2id) != len(bichar_id2elem):
        log=("serious bug: extern words dumplicated, please check!")
        print(log)
        log_.fprint_log(log)
    return (char_embeddings, char_elem2id, char_id2elem),(bichar_embeddings, bichar_elem2id, bichar_id2elem)
