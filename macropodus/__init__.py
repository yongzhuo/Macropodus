# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/12 22:57
# @author  : Mo
# @function: init of macropodus (tookit, keras of tensorflow)


# macropodus
from macropodus.tookit import calculate, chi2num, num2chi, Trie, roman2num, num2roman, pinyin, zh2han, han2zh
from macropodus.segment import cut_bidirectional, cut_forward, cut_reverse, cut_search, cut_dag, cut, find
from macropodus.segment import load_user_dict, save_delete_words, save_add_words, delete_word, add_word
from macropodus.summarize import keyword, textrank, summarization
from macropodus.__init_tf_keras import * # tf.python.keras, custom_objects
from macropodus.version import __version__ # 版本
from macropodus.similarity import sim

# 机械分词
cut_bidirectional = cut_bidirectional
cut_forward = cut_forward
cut_reverse = cut_reverse
cut_search = cut_search
cut_dag = cut_dag
cut = cut

# 用户词典操作
load_user_dict = load_user_dict
save_delete_words = save_delete_words # 保存到用户词典的
save_add_words = save_add_words
delete_word = delete_word
add_word = add_word

# 新词发现
find = find

# 文本相似度
sim = sim

# 文本摘要, 关键词
keyword = keyword
summarize = textrank
summarization = summarization

# 常用工具(tookit, 计算器, 中文与阿拉伯数字转化, 前缀树, 罗马数字与阿拉伯数字转化)
calculate = calculate
chi2num = chi2num
num2chi = num2chi
roman2num = roman2num
num2roman = num2roman
han2zh = han2zh
zh2han = zh2han
pinyin = pinyin
