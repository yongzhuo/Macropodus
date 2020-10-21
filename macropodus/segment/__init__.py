# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/18 22:00
# @author  : Mo
# @function: segment of sent


from macropodus.segment.seg_statistics.seg_statistics import SegStatistics
from macropodus.segment.word_discovery.word_discovery import WordDiscovery
import os


# 机械分词,默认使用缓存
use_cache = True
if not os.environ.get("macropodus_use_seg_cache", True):
    use_cache = False  # 不使用缓存，重新加载
segs = SegStatistics(use_cache)
cut_bidirectional = segs.cut_bidirectional
cut_forward = segs.cut_forward
cut_reverse = segs.cut_reverse
cut_search = segs.cut_search
cut_dag = segs.cut_dag
cut = segs.cut

# 用户词典增删改查
load_user_dict = segs.load_user_dict
save_delete_words = segs.save_delete_words
save_add_words = segs.save_add_words
delete_word = segs.delete_word
add_word = segs.add_word

# 新词发现
wd = WordDiscovery()
find = wd.find_word
