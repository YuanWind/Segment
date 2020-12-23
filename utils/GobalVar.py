# -*- coding: utf-8 -*-
# @Time    : 2020/12/23
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : global.py
def _init():
  global _global_dict
  _global_dict = {}


def set_value(name, value):
  _global_dict[name] = value


def get_value(name, defValue=None):
  try:
    return _global_dict[name]
  except KeyError:
    return defValue
