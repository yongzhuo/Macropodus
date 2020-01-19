# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/28 20:49
# @author  : Mo
# @function: tookit


# tookit
from macropodus.tookit.chinese2number.chinese2number import Chi2Num, Num2Chi
from macropodus.tookit.calculator_sihui.calcultor_sihui import Calculator
from macropodus.tookit.trie_tree.trie_tree import TrieTree
from macropodus.tookit.han2zh.han2zh import Han2Zh
from macropodus.tookit.pinyin.pinyin import PinYin
from macropodus.tookit.number2roman.ri import RI

# 常用工具(tookit, 计算器, 中文与阿拉伯数字转化, 前缀树, 中文与罗马数字相互转化, 中文转拼音, 繁简转化)
Calcul = Calculator()
Chi2num = Chi2Num()
Num2chi = Num2Chi()
Trie = TrieTree()
hanzh = Han2Zh()
piyi = PinYin()
ri = RI()

calculate = Calcul.calculator_sihui
chi2num = Chi2num.compose_decimal
num2chi = Num2chi.decimal_chinese
roman2num = ri.roman2int
num2roman = ri.int2roman
han2zh = hanzh.han2zh
zh2han = hanzh.zh2han
pinyin = piyi.pinyin
