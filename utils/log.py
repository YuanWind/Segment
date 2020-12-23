# -*- coding: utf-8 -*-
# @Time    : 2020/11/14
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : log.py
import os

class Log:
    def __init__(self, config):
        if config.log_file == '':
            raise RuntimeError('-log_fname must be given')
        self.path = os.path.join(config.log_dir,config.log_file)
        if not os.path.isdir(config.log_dir):
            os.mkdir(config.log_dir)
        self.config = config
        self.fprint_config()

    def fprint_config(self):
        with open(self.path, 'w', encoding='utf8') as f:
            f.write('config:\n')
            f.write('\n'.join(['%s:%s' % item for item in self.config.__dict__.items()]))
            f.write('\n\n')

    def fprint_log(self, text):
        with open(self.path, 'a', encoding='utf8') as f:
            f.write(text + '\n')

