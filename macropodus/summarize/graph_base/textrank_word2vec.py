# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/12/20 20:39
# @author   :Mo
# @function :textrank of word2vec, keyword and sentence


from macropodus.similarity.similarity_word2vec_char import SimW2vChar
from macropodus.data.words_common.stop_words import stop_words
from macropodus.preprocess.tools_ml import macropodus_cut
from macropodus.preprocess.tools_ml import cut_sentence
import networkx as nx
import numpy as np


class TextrankWord2vec(SimW2vChar):
    def __init__(self, use_cache=True):
        self.algorithm = 'textrank_word2vec'
        self.stop_words = stop_words
        super().__init__(use_cache) # self.w2v_char

    def cut_window(self, sent_words, win_size=2):
        """
            滑动窗口切词
        :param sent_words: list, like ["我", "是", "大漠帝国"]
        :param win_size: int, like 3
        :return: yield
        """
        if win_size < 2:
            win_size = 2
        for i in range(1, win_size):
            if i >= len(sent_words):
                break
            sent_terms = sent_words[i:] # 后面的
            sent_zip = zip(sent_words, sent_terms) # 候选词对
            for sz in sent_zip:
                yield sz

    def keyword(self, text, num=6, score_min=0.025, win_size=3, type_sim="total", type_encode="avg", config={"alpha": 0.86, "max_iter":100}):
        """
            关键词抽取, textrank of word2vec cosine
        :param text: str, doc. like "大漠帝国是历史上存在的国家吗?你知不知道？嗯。"
        :param num: int, length of sentence like 6
        :param win_size: int, windows size of combine. like 2
        :param type_sim: str, type of simiilarity. like "total", "cosine"
        :param config: dict, config of pagerank. like {"alpha": 0.86, "max_iter":100}
        :return: list, result of keyword. like [(0.020411696169510562, '手机'), (0.016149784106276977, '夏普')]
        """
        # 切句
        if type(text) == str:
            self.sentences = cut_sentence(text)
        elif type(text) == list:
            self.sentences = text
        else:
            raise RuntimeError("text type must be list or str")
        # macropodus_cut 切词
        self.macropodus_word = [macropodus_cut(sentence) for sentence in self.sentences]
        # 去除停用词等
        self.sentences_word = [[w for w in mw if w not in self.stop_words.values()] for mw in self.macropodus_word]
        # 构建图的顶点
        word2index = {}
        index2word = {}
        word_index = 0
        for sent_words in self.sentences_word:
            for word in sent_words:
                if not word in word2index: # index
                    word2index[word] = word_index
                    index2word[word_index] = word
                    word_index += 1
        graph_words = np.zeros((word_index, word_index))
        # 构建图的边, 以两个词语的余弦相似度为基础
        for sent_words in self.sentences_word:
            for cw_1, cw_2 in self.cut_window(sent_words, win_size=win_size):
                if cw_1 in word2index and cw_2 in word2index:
                    idx_1, idx_2 = word2index[cw_1], word2index[cw_2]
                    score_w2v_cosine = self.similarity(cw_1, cw_2, type_sim=type_sim,
                                                       type_encode=type_encode)
                    graph_words[idx_1][idx_2] = score_w2v_cosine
                    graph_words[idx_2][idx_1] = score_w2v_cosine
        # 构建相似度矩阵
        w2v_cosine_sim = nx.from_numpy_matrix(graph_words)
        # nx.pagerank
        sens_scores = nx.pagerank(w2v_cosine_sim, **config)
        # 得分排序
        sen_rank = sorted(sens_scores.items(), key=lambda x: x[1], reverse=True)
        # 保留topk个, 防止越界
        topk = min(len(sen_rank), num)
        # 返回原句子和得分
        return [(sr[1], index2word[sr[0]]) for sr in sen_rank if len(index2word[sr[0]])>1 and score_min<=sr[1]][0:topk]

    def summarize(self, text, num=320, type_sim="cosine", type_encode="avg", config={"alpha": 0.33, "max_iter":100}):
        """
            文本摘要抽取, textrank of word2vec cosine
        :param text: str, doc. like "大漠帝国是历史上存在的国家吗?你知不知道？嗯。"
        :param num: int, length of sentence like 6
        :param type_sim: str, type of simiilarity. like "total", "cosine"
        :param config: dict, config of pagerank. like {"alpha": 0.86, "max_iter":100}
        :return: list, result of keyword. like [(0.06900223298930287, 'PageRank The PageRank Citation Ranking'), (0.08698940285163381, 'PageRank通过网络浩瀚的超链接关系来确定一个页面的等级')]
        """
        # 切句
        if type(text) == str:
            self.sentences = cut_sentence(text)
        elif type(text) == list:
            self.sentences = text
        else:
            raise RuntimeError("text type must be list or str")
        # 输入文本句子长度
        len_sen = len(self.sentences)
        # 构建图的顶点
        sent2idx = {}
        idx2sent = {}
        sent_idx = 0
        for sent in self.sentences:
            sent2idx[sent] = sent_idx
            idx2sent[sent_idx] = sent
            sent_idx += 1
        graph_sents = np.zeros((sent_idx, sent_idx))
        # 构建图的边, 以两个句子的余弦相似度为基础
        for i in range(len_sen):
            for j in range(len_sen):
                score_w2v_cosine = self.similarity(self.sentences[i], self.sentences[j],
                                                   type_sim=type_sim, type_encode=type_encode)
                graph_sents[i][j] = score_w2v_cosine
                graph_sents[j][i] = score_w2v_cosine
        # 构建相似度矩阵
        w2v_cosine_sim = nx.from_numpy_matrix(graph_sents)
        # nx.pagerank
        sens_scores = nx.pagerank(w2v_cosine_sim, **config)
        # 得分排序
        sen_rank = sorted(sens_scores.items(), key=lambda x: x[1], reverse=True)
        # 保留topk个, 防止越界
        topk = min(len(sen_rank), num)
        # 返回原句子和得分
        return [(sr[1], self.sentences[sr[0]]) for sr in sen_rank][0:topk]


if __name__ == '__main__':
    text = "是上世纪90年代末提出的一种计算网页权重的算法。" \
          "当时，互联网技术突飞猛进，各种网页网站爆炸式增长，" \
          "业界急需一种相对比较准确的网页重要性计算方法，" \
          "是人们能够从海量互联网世界中找出自己需要的信息。" \
          "百度百科如是介绍他的思想:PageRank通过网络浩瀚的超链接关系来确定一个页面的等级。" \
          "Google把从A页面到B页面的链接解释为A页面给B页面投票，" \
          "Google根据投票来源甚至来源的来源，即链接到A页面的页面" \
          "和投票目标的等级来决定新的等级。简单的说，" \
          "一个高等级的页面可以使其他低等级页面的等级提升。" \
          "PageRank The PageRank Citation Ranking: Bringing Order to the Web，" \
          "具体说来就是，PageRank有两个基本思想，也可以说是假设，" \
          "即数量假设：一个网页被越多的其他页面链接，就越重）；" \
          "质量假设：一个网页越是被高质量的网页链接，就越重要。" \
          "总的来说就是一句话，从全局角度考虑，获取重要的信息。"
    trww = TextrankWord2vec()
    keyword = trww.keyword(text, num=8)
    summary = trww.summarize(text, num=32)
    print(keyword)
    print(summary)

