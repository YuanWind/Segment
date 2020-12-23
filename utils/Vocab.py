# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : Vocab.py
from collections import Counter

import torch
from tqdm import tqdm

from utils.Utils import reverse
from utils import GobalVar
log_=GobalVar.get_value('log_')
class Vocab:
    def __init__(self):

        self.id2word={0:'<padw>',1:'<unkw>'}     # 最后一个字母 w 指的是 word
        self.word2id =None
        self.wordid2freq={0:10000,1:10000}

        self.id2char={0:'<padc>',1:'<unkc>'}     # 最后一个字母 c 指的是 char
        self.char2id =None

        self.id2bichar={0:'<padbc>',1:'<unkbc>'} # 最后两个字母 bc 指的是 bi-char
        self.bichar2id =None

        self.exchar2id=None
        self.exbichar2id=None

        self.exid2char=None
        self.exid2bichar=None

        self.id2label = {}
        self.label2id = None

    def build(self, data, min_occur_cnt=2):
        word_counter = Counter()
        bichar_counter = Counter()
        char_counter = Counter()
        for inst in data:
            for w in inst.words:
                word_counter[w] += 1
            for c in inst.chars:
                char_counter[c] += 1
            for bc in inst.bichars:
                bichar_counter[bc] += 1
        for word, count in word_counter.most_common():
            if count > min_occur_cnt:
                self.id2word[len(self.id2word)] = word
                self.wordid2freq[len(self.wordid2freq)] = word

        for char, count in char_counter.most_common():
            self.id2char[len(self.id2char)] = char

        for bichar, count in bichar_counter.most_common():
            self.id2bichar[len(self.id2bichar)] = bichar

        self.word2id = reverse(self.id2word)
        if len(self.word2id) != len(self.id2word):
            print("serious bug: words dumplicated, please check!")

        self.char2id = reverse(self.id2char)
        if len(self.char2id) != len(self.id2char):
            log=("serious bug: chars dumplicated, please check!")
            print(log)
            log_.fprint_log(log)
        self.bichar2id = reverse(self.id2bichar)
        if len(self.bichar2id) != len(self.id2bichar):
            log=("serious bug: bichars dumplicated, please check!")
            print(log)
            log_.fprint_log(log)

        log=("Vocab info: #char %d, #bichar %d" % (self.char_size, self.bichar_size))
        print(log)
        log_.fprint_log(log)
    def load_pretrained_char_vec(self,emb_file):
        embedding_dim = -1
        word_count = self.char_size
        with open(emb_file, encoding='utf-8') as f:
            line = f.readline()  # 第一行是相关信息，所以要接着读取一行
            line = f.readline()
            line = line.strip().split()
            if len(line) <= 1:
                log="load_predtrained_embedding text is wrong!  -> len(line) <= 1"
                print(log)
                log_.fprint_log(log)
            else:
                embedding_dim = len(line) - 1
        embeddings = torch.zeros([word_count, embedding_dim])
        cnt=0
        oov=torch.zeros([1,embedding_dim])
        oov_cnt=0
        with open(emb_file, encoding='utf-8') as f:
            i=0
            for line in tqdm(f):
                if i==0:
                    i+=1
                    continue
                line=line.replace('\n','')
                cnt+=1
                values = line.split(' ')
                word = values[0]
                vector = torch.tensor([float(i) for i in values[1:-1]])
                if word in self.char2id.keys():
                    embeddings[self.char2id[word]]=vector
                    embeddings[self.char2id['<unkc>']]+=vector
                oov+=vector
        embeddings[self.char2id['<unkc>']] = embeddings[self.char2id['<unkc>']] / cnt
        for idx in range(word_count): # 处理 训练集字符没有在预训练词典中出现的情况
            x=embeddings[idx]
            y=torch.zeros([1,embedding_dim])
            if idx>1 and 0 == ((x != y).sum()):
                embeddings[idx]=oov
                oov_cnt+=1

        log=('{} is not in pretrained vec!'.format(oov_cnt))
        print(log)
        log_.fprint_log(log)
        embeddings = embeddings / torch.std(embeddings)
        return embeddings,embedding_dim

    def create_label(self, data):
        label_counter = Counter()
        for sample in data:
            for label in sample.char_labels:
                label_counter[label] += 1

        for label, count in label_counter.most_common():
            self.id2label[len(self.id2label)]=label
        self.label2id = reverse(self.id2label)
        if len(self.label2id) != len(self.id2label):
            log=("serious bug: label dumplicated, please check!")
            print(log)
            log.fprint_log(log)
    @property
    def char_size(self):
        return len(self.id2char)

    @property
    def exbichar_size(self):
        return len(self.exid2bichar)

    @property
    def exchar_size(self):
        return len(self.exid2char)
    @property
    def bichar_size(self):
        return len(self.id2bichar)
    @property
    def label_size(self):
        return len(self.id2label)
