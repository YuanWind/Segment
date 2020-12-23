# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : YYWind
# @Email   : yywind@126.com
# @File    : Preprocess.py
import re
def number2ch(text,ch='<num>'):
    """将数字转为标记 <num>, 注意：如果出现电话、身份证号也会转为标记"""
    digits = re.findall(r"\d+", text)
    digits.sort(key=lambda i: len(i), reverse=True)
    for digit in digits:
        text = text.replace(digit, ch)
    return text

def del_blank(text):
    """删除各种空格符和制表符"""
    text = text.replace(" ", "") # unicode:8195
    text = text.replace(' ', "")    # unicode:32
    text = text.replace("　", "")   # unicode:12288
    text = text.replace("\t", "")   # unicode:9
    return text

def url2ch(text,ch='<url>'):
    """将 url 链接转为标记 <url> """
    results = re.compile(r'http://[a-zA-Z0-9.?/&=\-:]*', re.S)
    text = results.sub(ch, text)
    results = re.compile(r'https://[a-zA-Z0-9.?/&=\-:]*', re.S)
    text = results.sub(ch, text)
    return text

def del_odd_ch(text):
    """清除除中英文字符，标点，数字之外的符号"""
    cop = re.compile("[^\u4e00-\u9fa5^.^a-z^A-Z^0-9.\W]")
    text = cop.sub("", text)
    return text

def del_html_label(text):
    """去掉html标签"""
    from bs4 import BeautifulSoup
    text = BeautifulSoup(text, 'html.parser').get_text()
    return text

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def char2lower(text):
    """大小写统一"""
    text = text.lower()
    return text

def del_stop_words(stop_words=[]):
    """删除停用词"""
    pass
def chinese_text_process(text):
    text=del_odd_ch(text)
    text=strQ2B(text)
    text=url2ch(text)
    text=number2ch(text)
    text=del_blank(text)
    text=del_html_label(text)
    text=char2lower(text)
    return text
