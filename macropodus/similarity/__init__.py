# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/18 22:04
# @author  : Mo
# @function:


from macropodus.similarity.similarity_word2vec_char import SimW2vChar
import os

# 词向量, 默认使用缓存
use_cache = True
if not os.environ.get("macropodus_use_w2v_cache", True):
    use_cache = False  # 不使用缓存，重新加载
# 文本相似度
swc = SimW2vChar(use_cache)
sim = swc.similarity
