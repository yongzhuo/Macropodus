# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/5 22:52
# @author  : Mo
# @function: word2vec of gensim


from macropodus.conf.path_config import path_embedding_word2vec_char, path_macropodus_w2v_char_cache
from macropodus.conf.path_log import get_logger_root
import numpy as np
import gensim
import pickle
import time
import os


logger = get_logger_root()
gensim.logger.level=40


class W2v:
    def __init__(self, use_cache=True):
        # time_start = time.time()
        # 存在缓存则直接读取, 序列化加速缓存读取速度
        if use_cache and os.path.exists(path_macropodus_w2v_char_cache):
            with open(path_macropodus_w2v_char_cache, "rb") as fpmc:
                self.w2v_char= pickle.load(fpmc)
                fpmc.close()
                # logger.info("word2vec: " + str(time.time() - time_start)) # 0.12
        else:
            # gensim加载词向量
            self.w2v_char = gensim.models.KeyedVectors.load_word2vec_format(path_embedding_word2vec_char)
            # logger.info("word2vec: " + str(time.time() - time_start)) # 0.99, 0.78
            # 第一次跑macropodus, 序列化需要的缓存
            if use_cache and not os.path.exists(path_macropodus_w2v_char_cache):
                with open(path_macropodus_w2v_char_cache, "wb") as fpmc:
                    pickle.dump(self.w2v_char, fpmc)

    def cosine(self, sen_1, sen_2):
        """
            余弦距离
        :param sen_1: numpy.array
        :param sen_2: numpy.array
        :return: float, like 0.0
        """
        if sen_1.all() and sen_2.all():
            return np.dot(sen_1, sen_2) / (np.linalg.norm(sen_1) * np.linalg.norm(sen_2))
        else:
            return 0.0

    def jaccard(self, sen_1, sen_2):
        """
            jaccard距离
        :param sen1: str, like "大漠帝国"
        :param sen2: str, like "Macropodus"
        :return: float, like 0.998
        """
        try:
            sent_intersection = list(set(list(sen_1)).intersection(set(list(sen_2))))
            sent_union = list(set(list(sen_1)).union(set(list(sen_2))))
            score_jaccard = float(len(sent_intersection) / len(sent_union))
        except:
            score_jaccard = 0.0
        return score_jaccard
