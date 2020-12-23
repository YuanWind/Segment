# -*- coding: utf-8 -*-
# @Time    : 2020/12/20
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : Trainer.py
import itertools
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from modules.Optimizer import Optimizer
from utils.Evaluate import evaluate
from utils.Utils import check_exist
from utils import GobalVar
log_=GobalVar.get_value('log_')

def label2variable(onebatch, vocab):
    batch_size = len(onebatch)
    lengths = []
    for inst in onebatch:
        lengths.append(len(inst.chars))
    max_len = max(lengths)
    batch_char_labels = np.ones((batch_size, max_len), dtype=int)

    for idx in range(batch_size):
        char_label_strs = onebatch[idx].char_labels
        char_label_indexes = [vocab.label2id.get(i) for i in char_label_strs]
        idy = 0
        for label_index in char_label_indexes:
            batch_char_labels[idx][idy] = label_index
            idy += 1
    batch_char_labels = torch.tensor(batch_char_labels, dtype=torch.long)
    return batch_char_labels

def data2variable(onebatch, vocab):
    batch_size = len(onebatch)
    char_lengths = []
    bichar_lengths = []
    for inst in onebatch:
        assert len(inst.chars) == len(inst.bichars)
        char_lengths.append(len(inst.chars))
        bichar_lengths.append(len(inst.bichars))

    max_char_len = max(char_lengths)
    max_bichar_len = max(bichar_lengths)

    batch_chars = torch.zeros((batch_size, max_char_len), dtype=torch.long)
    batch_bichars = torch.zeros((batch_size, max_bichar_len), dtype=torch.long)
    batch_exchars = torch.zeros((batch_size, max_char_len), dtype=torch.long)
    batch_exbichars = torch.zeros((batch_size, max_bichar_len), dtype=torch.long)
    char_mask = torch.zeros((batch_size, max_char_len), dtype=torch.float)

    for idx in range(batch_size):
        char_indexes = [vocab.char2id.get(i,vocab.char2id.get('<unkc>')) for i in onebatch[idx].chars]
        bichar_indexes = [vocab.bichar2id.get(i,vocab.bichar2id.get('<unkbc>')) for i in onebatch[idx].bichars]
        exchar_indexes = [vocab.exchar2id.get(i,vocab.exchar2id.get('<unkc>')) for i in onebatch[idx].chars]
        exbichar_indexes = [vocab.exbichar2id.get(i,vocab.exbichar2id.get('<unkbc>')) for i in onebatch[idx].bichars]

        for idy, char_idx in enumerate(char_indexes):
            batch_chars[idx,idy] = char_idx
            char_mask[idx,idy] = 1
        for idy, bichar_idx in enumerate(bichar_indexes):
            batch_bichars[idx,idy] = bichar_idx
        for idy,exchar_idx in enumerate(exchar_indexes):
            batch_exchars[idx,idy]=exchar_idx
        for idy,exbichar_idx in enumerate(exbichar_indexes):
            batch_exbichars[idx,idy]=exbichar_idx
    label_mask=char_mask.type(torch.long)
    return batch_chars,batch_bichars,batch_exchars,batch_exbichars,char_mask, label_mask

def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]
        yield sentences

def data_iter(data, batch_size, shuffle=True):

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def train(train_data, dev_data,test_data, vocab, config, model):
    model_param = filter(lambda p: p.requires_grad,
                             itertools.chain(model.lstm.parameters(),model.crf.parameters())
                        )
    model_optimizer = Optimizer(model_param, config)

    batch_num = int(np.ceil(len(train_data) / float(config.train_batch_size)))
    global_step = 0
    best_F = 0
    for iter in range(config.train_iters):
        start_time = time.time()
        # print('Iteration: ' + str(iter))
        batch_iter = 0
        overall_correct,  overall_total = 0, 0
        for onebatch in data_iter(train_data, config.train_batch_size, True):
            batch_char_labels = label2variable(onebatch, vocab)
            batch_chars,batch_bichars,batch_exchars,batch_exbichars,char_mask, label_mask=\
                data2variable(onebatch, vocab)
            model.train()

            model.forward(batch_chars,batch_bichars,batch_exchars,batch_exbichars,char_mask)
            loss = model.compute_crf_loss(batch_char_labels, label_mask)

            total, correct = model.compute_acc(batch_char_labels, label_mask)
            overall_total += total
            overall_correct += correct
            acc = overall_correct / overall_total
            loss_value = loss.data.cpu().numpy()
            loss.backward()

            during_time = float(time.time() - start_time)
            log=("Step:%d, epoch:%d, batch:%d, time:%.2f, acc:%.2f, loss:%.2f"
                  %(global_step, iter, batch_iter,  during_time, acc, loss_value))
            print(log)
            log_.fprint_log(log)
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm_(model_param, max_norm=config.clip)

                model_optimizer.step()
                model_optimizer.zero_grad()

                global_step += 1

        if iter % config.validate_every == 0:
            segment(dev_data, model, vocab, config, config.data_dir+config.dev_file+'.dev.tmp')
            dev_seg_eval = evaluate(config.data_dir+config.dev_file,config.data_dir+config.dev_file+'.dev.tmp')

            log=("Dev:")
            print(log)
            log_.fprint_log(log)
            dev_seg_eval.print()
            segment(test_data, model, vocab, config, config.data_dir+config.test_file + '.test.tmp')
            test_seg_eval = evaluate(config.data_dir+config.test_file, config.data_dir+config.test_file +'.test.tmp')
            log("Test:")
            print(log)
            log_.fprint_log(log)
            test_seg_eval.print()
            dev_F = dev_seg_eval.getAccuracy()
            if best_F < dev_F:
                log=("Exceed best Full F-score: history = %.2f, current = %.2f" % (best_F, dev_F))
                print(log)
                log_.fprint_log(log)
                best_F = dev_F

                if config.save_after >= 0 and iter >= config.save_after:
                    log=("Save model")
                    print(log)
                    log_.fprint_log(log)
                    model_params = {'lstm': model.lstm.state_dict(),
                                    'crf': model.crf.state_dict()}
                    torch.save(model_params, check_exist(config.save_dir,config.best_model_file))
def gen_test_res(test_data, vocab, config, model):
    model.load_stict()
    segment(test_data, model, vocab, config, config.data_dir + config.test_file + '.test.result.txt')
    test_seg_eval = evaluate(config.data_dir + config.test_file, config.data_dir + config.test_file + '.test.result.txt')
    log=("Test:")
    print(log)
    log_.fprint_log(log)
    test_seg_eval.print()
def segment(data, model, vocab, config, outputFile, split_str=' '):
    start = time.time()
    outf = open(outputFile, mode='w', encoding='utf8')
    for onebatch in tqdm(data_iter(data, config.test_batch_size, False)):
        b = len(onebatch)
        seg = False
        for idx in range(b):
            if len(onebatch[idx].chars) > 0:
                seg = True
                break
        if seg:
            batch_chars,batch_bichars,batch_exchars,batch_exbichars,char_mask, label_mask= \
                data2variable(onebatch, vocab)
            model.eval()
            model.forward(batch_chars,batch_bichars,batch_exchars,batch_exbichars,char_mask)
            best_paths = model.viterbi_decode(label_mask) # 维比特算法
            labels = path2labels(best_paths, vocab)
            outputs = labels2output(onebatch, labels)
            for sent in outputs:
                outf.write(split_str.join(sent) + '\n')
        else:
            for idx in range(b):
                outf.write('\n')
    during_time = float(time.time() - start)
    outf.close()
    log=("sentence num: %d,  segment time = %.2f " % (len(data), during_time))
    print(log)
    log_.fprint_log(log)


def path2labels(paths, vocab):
    if isinstance(paths, list):
        return [[vocab.id2label.get(i) for i in x] for x in paths]
    return vocab.id2label.get(paths)

def labels2output(onebatch, labels):
    outputs = []
    for idx, inst in enumerate(onebatch):
        predict_labels = labels[idx]
        assert len(predict_labels) == len(inst.chars)

        tmp = ''
        predict_sent = []
        for idy, label in enumerate(predict_labels):
            if label == 'b':
                if idy > 0:
                    predict_sent.append(tmp)
                tmp = inst.chars[idy]

            if label == 'i':
                tmp += inst.chars[idy]
        if tmp is not '':
            predict_sent.append(tmp)
        outputs.append(predict_sent)

    return outputs
