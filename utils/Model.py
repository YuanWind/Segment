# -*- coding: utf-8 -*-
# @Time    : 2020/12/20
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : Model.py
import torch
from modules.BiLSTM import BiLSTM
from modules.MyCRF import MyCRF
from utils.Utils import load_pkl_data, split_pretrained_vec, dump_pkl_data
from allennlp.modules.conditional_random_field import ConditionalRandomField as CRF


class Model:
    def __init__(self):
        self.lstm=None
        self.crf=None
        self.use_cuda=None
        self.best_model=None

    def build(self,vocab,config):

        #读取预训练向量，用pickle序列化保存起来，方便下次直接使用
        # (char_embeddings, char_elem2id, char_id2elem),\
        # (bichar_embeddings, bichar_elem2id, bichar_id2elem)=\
        #     split_pretrained_vec(config.pretrained_embedding_file)
        # dump_pkl_data((char_embeddings, char_elem2id, char_id2elem), config.save_dir+'char_embed_no_std.pkl')
        # dump_pkl_data((bichar_embeddings, bichar_elem2id, bichar_id2elem), config.save_dir+'bichar_embed_no_std.pkl')

        (exchar_embeddings, vocab.exchar2id, vocab.exid2char) = load_pkl_data(config.save_dir + 'char_embed_no_std.pkl')
        (exbichar_embeddings, vocab.exbichar2id, vocab.exid2bichar) = load_pkl_data(
            config.save_dir + 'bichar_embed_no_std.pkl')
        char_embedding_dim = len(exchar_embeddings[0])
        bichar_embedding_dim = len(exbichar_embeddings[0])

        # 纠正 config 中的嵌入维度
        config.char_dims = char_embedding_dim
        config.bichar_dims = bichar_embedding_dim
        self.lstm = BiLSTM(vocab, config)
        self.lstm.init_pretrain_vec(exchar_embeddings, exbichar_embeddings)

        if config.use_own:
            self.crf = MyCRF(num_tags=vocab.label_size,
                           constraints=None,
                           include_start_end_transitions=False)
        else:
            self.crf = CRF( num_tags=vocab.label_size,
                            constraints=None,
                            include_start_end_transitions=False)
        self.best_model=config.save_dir + config.best_model_file
        if config.gpu:
            self.use_cuda=True
        else:
            self.use_cuda=False
        if self.use_cuda:
            torch.backends.cudnn.enabled = True
            self.lstm = self.lstm.cuda()
            self.crf = self.crf.cuda()
    def train(self):
        if self.lstm is not None:
            self.lstm.train()
        if self.crf is not None:
            self.crf.train()
        self.training = True

    def eval(self):
        if self.lstm is not None:
            self.lstm.eval()
        if self.crf is not None:
            self.crf.eval()
        self.training = False

    def forward(self, batch_chars,batch_bichars,batch_exchars,batch_exbichars,char_mask):
        if self.use_cuda:
            batch_chars = batch_chars.cuda()
            batch_bichars = batch_bichars.cuda()
            batch_exchars = batch_exchars.cuda()
            batch_exbichars = batch_exbichars.cuda()
            char_mask = char_mask.cuda()
        self.logit = self.lstm(batch_chars,batch_bichars,batch_exchars,batch_exbichars,char_mask)

    def load_stict(self):
        model_params=torch.load(self.best_model)
        if self.lstm is not None:
            self.lstm.load_state_dict(model_params['lstm'])
        if self.crf is not None:
            self.crf.load_state_dict(model_params['crf'])

    def viterbi_decode(self, labels_mask):
        if self.use_cuda:
            labels_mask=labels_mask.cuda()
        # self.logit: [batch_size,seq_len,label_cnt] [32,150,2]
        output = self.crf.viterbi_tags(self.logit, labels_mask)
        best_paths = []
        for path, score in output:
            best_paths.append(path)
        return best_paths

    def compute_crf_loss(self, char_labels, labels_mask):
        if self.use_cuda:
            char_labels = char_labels.cuda()
            labels_mask = labels_mask.cuda()
        b = char_labels.size(0)
        crf_loss = -self.crf(self.logit, char_labels, labels_mask) / b ##
        return crf_loss

    def compute_acc(self, char_labels, labels_mask):
        b, seq_len = labels_mask.size()
        true_lengths = torch.sum(labels_mask, dim=1).numpy()
        pred_labels = self.logit.data.max(2)[1].cpu().numpy()
        char_labels = char_labels.cpu().numpy()
        correct = 0
        total = 0
        for idx in range(b):
            true_len = true_lengths[idx]
            total += true_len
            for idy in range(true_len):
                if pred_labels[idx][idy] == char_labels[idx][idy]:
                    correct += 1
        return total, correct