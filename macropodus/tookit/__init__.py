# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/28 20:49
# @author  : Mo
# @function:


# tookit
from macropodus.tookit.chinese2number.chinese2number import Chi2Num, Num2Chi
from macropodus.tookit.calculator_sihui.calcultor_sihui import Calculator
from macropodus.tookit.trie_tree.trie_tree import TrieTree

# 常用工具(tookit, 计算器, 中文与阿拉伯数字转化, 前缀树)
Calcul = Calculator()
Chi2num = Chi2Num()
Num2chi = Num2Chi()
Trie = TrieTree()
calculate = Calcul.calculator_sihui
chi2num = Chi2num.compose_decimal
num2chi = Num2chi.decimal_chinese
