# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/11/28 20:16
# @author   :Mo
# @function :mmr


from macropodus.preprocess.tools_ml import extract_chinese, cut_sentence
from macropodus.preprocess.tools_ml import macropodus_cut, tfidf_fit
from macropodus.data.words_common.stop_words import stop_words
import copy


class MMRSum:
    def __init__(self):
        self.stop_words = stop_words.values()
        self.algorithm = 'mmr'

    def summarize(self, text, num=8, alpha=0.6):
        """

        :param text: str
        :param num: int
        :return: list
        """
        # 切句
        if type(text) == str:
            self.sentences = cut_sentence(text)
        elif type(text) == list:
            self.sentences = text
        else:
            raise RuntimeError("text type must be list or str")
        # 切词
        sentences_cut = [[word for word in macropodus_cut(extract_chinese(sentence))
                          if word.strip()] for sentence in self.sentences]
        # 去除停用词等
        self.sentences_cut = [list(filter(lambda x: x not in self.stop_words, sc)) for sc in sentences_cut]
        self.sentences_cut = [" ".join(sc) for sc in self.sentences_cut]
        # # 计算每个句子的词语个数
        # sen_word_len = [len(sc)+1 for sc in sentences_cut]
        # 计算每个句子的tfidf
        sen_tfidf = tfidf_fit(self.sentences_cut)
        # 矩阵中两两句子相似度
        SimMatrix = (sen_tfidf * sen_tfidf.T).A # 例如: SimMatrix[1, 3]  # "第2篇与第4篇的相似度"
        # 输入文本句子长度
        len_sen = len(self.sentences)
        # 句子标号
        sen_idx = [i for i in range(len_sen)]
        summary_set = []
        mmr = {}
        for i in range(len_sen):
            if not self.sentences[i] in summary_set:
                sen_idx_pop = copy.deepcopy(sen_idx)
                sen_idx_pop.pop(i)
                # 两两句子相似度
                sim_i_j = [SimMatrix[i, j] for j in sen_idx_pop]
                score_tfidf = sen_tfidf[i].toarray()[0].sum() # / sen_word_len[i], 如果除以词语个数就不准确
                mmr[self.sentences[i]] = alpha * score_tfidf - (1 - alpha) * max(sim_i_j)
                summary_set.append(self.sentences[i])
        score_sen = [(rc[1], rc[0]) for rc in sorted(mmr.items(), key=lambda d: d[1], reverse=True)]
        if len(mmr) > num:
            score_sen = score_sen[0:num]
        return score_sen


if __name__ == '__main__':
    mmr_sum = MMRSum()
    doc = "PageRank算法简介。" \
          "是上世纪90年代末提出的一种计算网页权重的算法! " \
          "当时，互联网技术突飞猛进，各种网页网站爆炸式增长。 " \
          "业界急需一种相对比较准确的网页重要性计算方法。 " \
          "是人们能够从海量互联网世界中找出自己需要的信息。 " \
          "百度百科如是介绍他的思想:PageRank通过网络浩瀚的超链接关系来确定一个页面的等级。 " \
          "Google把从A页面到B页面的链接解释为A页面给B页面投票。 " \
          "Google根据投票来源甚至来源的来源，即链接到A页面的页面。 " \
          "和投票目标的等级来决定新的等级。简单的说， " \
          "一个高等级的页面可以使其他低等级页面的等级提升。 " \
          "具体说来就是，PageRank有两个基本思想，也可以说是假设。 " \
          "即数量假设：一个网页被越多的其他页面链接，就越重）。 " \
          "质量假设：一个网页越是被高质量的网页链接，就越重要。 " \
          "总的来说就是一句话，从全局角度考虑，获取重要的信。 "
    sum = mmr_sum.summarize(doc)
    for i in sum:
        print(i)






