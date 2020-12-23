from utils.Metric import *

def get_ent(words):
    start = 0
    ents = []
    for w in words:
        ent_str = '[' + str(start) + ',' + str(start + len(w) - 1) + ']'
        ents.append(ent_str)
        start += len(w)
    return ents

def check(words1, words2):
    str1 = ''
    for w in words1:
        str1 += w
    str2 = ''
    for w in words2:
        str2 += w
    assert str1 == str2

def evaluate(true_file, predict_file):
    metric = Metric()
    t_inf = open(true_file, mode='r', encoding='utf8')
    p_inf = open(predict_file, mode='r', encoding='utf8')

    predict_num = 0
    correct_num = 0
    true_num = 0

    for t_line, p_line in zip(t_inf.readlines(), p_inf.readlines()):
        t_words = t_line.strip().split(" ")
        p_words = p_line.strip().split(" ")
        check(t_words, p_words)
        true_set = set(get_ent(t_words))
        predict_set = set(get_ent(p_words))
        predict_num += len(predict_set)
        true_num += len(true_set)
        correct_num += len(predict_set & true_set)

    metric.TP = correct_num
    metric.TP_FP = predict_num
    metric.TP_FN = true_num

    t_inf.close()
    p_inf.close()

    return metric