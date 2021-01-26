# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : main.py
from utils import GobalVar
from utils.log import Log
import torch
class Config:
    def __init__(self):
        # [Data]
        self.pretrained_embedding_file='H://competition/pre_embed/sgns.target.word-ngram.1-2.dynwin5.thr10.neg5.dim300.iter5'
        # self.data_dir = 'data/ctb/'
        # self.train_file = 'sample.txt'
        # self.dev_file = 'sample.txt'
        # self.test_file = 'sample.txt'
        self.data_dir = 'data/ctb60/'
        self.train_file = 'train.ctb60.seg.hwc'
        self.dev_file = 'dev.ctb60.seg.hwc'
        self.test_file = 'test.ctb60.seg.hwc'
        self.min_occur_cnt=0
        self.use_own=False
        # [Saved]
        self.save_dir='saved/'
        self.log_dir='log/'
        self.log_file='log.txt'
        self.best_model_file= 'best_model.model'
        self.save_vocab_file='vocab.vocab'
        self.save_after=10
        # [Network]
        self.lstm_layers=1
        self.lstm_hiddens=200
        self.char_dims=300
        self.bichar_dims=300
        self.input_dims=300
        self.dropout_emb=0.33
        self.dropout_lstm_input=0.33
        self.dropout_lstm_hidden=0.33
        self.dropout_mlp=0

        # [Optimizer]
        self.optim='Adam'
        self.learning_rate=2e-3
        self.decay= 0.75
        self.decay_steps=500
        self.beta_1= 0.9
        self.beta_2= 0.9
        self.epsilon=1e-12
        self.clip=5.0

        # [Run]
        self.gpu=True
        self.num_buckets_train=40
        self.num_buckets_valid=10
        self.num_buckets_test=10
        self.train_iters=50
        self.train_batch_size=32
        self.test_batch_size=32
        self.validate_every=1
        self.update_every=4
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

