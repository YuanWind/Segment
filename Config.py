# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : Config.py
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
        self.gpu=False
        self.num_buckets_train=40
        self.num_buckets_valid=10
        self.num_buckets_test=10
        self.train_iters=50
        self.train_batch_size=32
        self.test_batch_size=32
        self.validate_every=1
        self.update_every=4