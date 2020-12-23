# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : Dataloader.py
from utils.Sample import Sample

def parse_sent(text):
    words=text.split(' ')
    chars=[]
    bichars=[]
    sample=Sample()
    for w in words:
        for idx,ch in enumerate(w):
            chars.append(ch)
    for idx in range(len(chars)):
        if idx==0:
            bichars.append('-NULL-'+chars[idx])
        else:
            bichars.append(chars[idx-1]+chars[idx])
    sample.chars=chars
    sample.bichars=bichars
    sample.words=words
    return sample
def read_corpus(file_path):
    data=[]
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            sample=parse_sent(line)
            data.append(sample)
    return data

def add_label(data):
    """给数据集加标签"""
    for sample in data:
        sample.char_labels=[]
        for w in sample.words:
            for idx,ch in enumerate(w):
                if idx==0:
                    sample.char_labels.append('b')
                else:
                    sample.char_labels.append('i')
        assert len(sample.char_labels)==len(sample.chars)
