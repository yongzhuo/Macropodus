# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/8/21 22:01
# @author   :Mo
# @function : textrank using tfidf of sklearn, pagerank of networkx


from sklearn.feature_extraction.text import TfidfTransformer
from macropodus.preprocess.tools_ml import cut_sentence
from macropodus.preprocess.tools_ml import tdidf_sim
import networkx as nx


class TextrankSklearn:
    def __init__(self):
        self.algorithm = 'textrank_sklearn'

    def summarize(self, text, num=320):
        # 切句
        if type(text) == str:
            sentences = cut_sentence(text)
        elif type(text) == list:
            sentences = text
        else:
            raise RuntimeError("text type must be list or str")
        # tf-idf相似度
        matrix = tdidf_sim(sentences)
        matrix_norm = TfidfTransformer().fit_transform(matrix)
        # 构建相似度矩阵
        tfidf_sim = nx.from_scipy_sparse_matrix(matrix_norm * matrix_norm.T)
        # nx.pagerank
        sens_scores = nx.pagerank(tfidf_sim)
        # 得分排序
        sen_rank = sorted(sens_scores.items(), key=lambda x: x[1], reverse=True)
        # 保留topk个, 防止越界
        topk = min(len(sentences), num)
        # 返回原句子和得分
        return [(sr[1], sentences[sr[0]]) for sr in sen_rank][0:topk]


if __name__ == '__main__':
    doc = "是上世纪90年代末提出的一种计算网页权重的算法。" \
          "当时，互联网技术突飞猛进，各种网页网站爆炸式增长，" \
          "业界急需一种相对比较准确的网页重要性计算方法，" \
          "是人们能够从海量互联网世界中找出自己需要的信息。" \
          "百度百科如是介绍他的思想:PageRank通过网络浩瀚的超链接关系来确定一个页面的等级。" \
          "Google把从A页面到B页面的链接解释为A页面给B页面投票，" \
          "Google根据投票来源甚至来源的来源，即链接到A页面的页面" \
          "和投票目标的等级来决定新的等级。简单的说，" \
          "一个高等级的页面可以使其他低等级页面的等级提升。" \
          "PageRank The PageRank Citation Ranking: Bringing Order to the Web，"\
          "具体说来就是，PageRank有两个基本思想，也可以说是假设，" \
          "即数量假设：一个网页被越多的其他页面链接，就越重）；" \
          "质量假设：一个网页越是被高质量的网页链接，就越重要。" \
          "总的来说就是一句话，从全局角度考虑，获取重要的信息。"
    doc = doc.encode('utf-8').decode('utf-8')
    ts = TextrankSklearn()
    textrank_tfidf = ts.summarize(doc, 32)
    for score_sen in textrank_tfidf:
        print(score_sen)
