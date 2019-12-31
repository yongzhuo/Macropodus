# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/18 22:04
# @author  : Mo
# @function:


from macropodus.similarity.similarity_word2vec_char import SimW2vChar


# 文本相似度
use_cache = True # 使用缓存
swc = SimW2vChar(use_cache)
sim = swc.similarity
