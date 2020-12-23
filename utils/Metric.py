# -*- coding: utf-8 -*-
# @Time    : 2020/12/20
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : Metric.py
from utils import GobalVar
log_=GobalVar.get_value('log_')
class Metric:
    def __init__(self):
        self.epsilon=1e-12
        self.TP_FN = 0 # 实际 1 的样本数量
        self.TP = 0    # 预测为 1 且实际也为 1 的样本数量
        self.TP_FP = 0 # 预测为 1 的样本数量

    def reset(self):
        self.TP_FN = 0
        self.TP = 0
        self.TP_FP = 0

    def getPrecision(self):
        return self.TP*1.0 / (self.TP_FP+self.epsilon)

    def getRecall(self):
        return self.TP*1.0 / (self.TP_FN+self.epsilon)

    def getF1(self):
        return self.getPrecision()*self.getRecall()*2.0 / (self.getPrecision() + self.getRecall())

    def print(self):
        if self.TP == 0:
            log=("Fmeasure: 0.0")
            print(log)
            log_.fprint_log(log)
        else:
            log1=("Recall: " + str(self.TP) + "/" + str(self.TP_FN) + "={:.6f}; ".format(self.getRecall()))
            log2=("Precision: " + str(self.TP) + "/" + str(self.TP_FP) + "={:.6f}; " .format(self.getPrecision()))
            log3=("Fmeasure: {:.6f}".format(self.getF1()))
            print(log1+log2+log3)
            log_.fprint_log(log1+log2+log3)