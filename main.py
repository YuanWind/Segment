# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : main.py
from Config import Config
from utils import GobalVar
from utils.log import Log
import torch
GobalVar._init()
config=Config()
log_=Log(config)
GobalVar.set_value('log_', log_)
if config.gpu:
    gpu = torch.cuda.is_available()
    log1 = ("GPU available: {} ".format(gpu))
    log2 = (" CuDNN: {}".format(torch.backends.cudnn.enabled))
    print(log1, log2)
    log_.fprint_log(log1 + log2)
from utils.Dataloader import *
from utils.Model import Model
from utils.Trainer import train, gen_test_res
from utils.Utils import *
from utils.Vocab import *
if __name__ == '__main__':
    set_seed()

    train_data=read_corpus(config.data_dir+config.train_file)
    dev_data=read_corpus(config.data_dir+config.dev_file)
    test_data=read_corpus(config.data_dir+config.test_file)

    add_label(train_data)
    add_label(dev_data)
    add_label(test_data)

    log='train num: {}; dev num: {}; test num: {}'.format(len(train_data),len(dev_data),len(test_data))
    print(log)
    log_.fprint_log(log)

    # 用训练集数据创建词表 vocab
    vocab=Vocab()
    vocab.build(train_data,config.min_occur_cnt)
    vocab.create_label(train_data)

    # 创建模型

    model=Model()
    model.build(vocab,config)

    train(train_data, dev_data,test_data,  vocab, config, model)
    gen_test_res(test_data,vocab, config, model)

