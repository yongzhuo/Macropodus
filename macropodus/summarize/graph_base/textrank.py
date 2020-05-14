# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/11/29 22:39
# @author   :Mo
# @function :textrank of textrank4zh, sklearn or gensim


from macropodus.summarize.graph_base.textrank_word2vec import TextrankWord2vec
from macropodus.summarize.graph_base.textrank_gensim import TextrankGensimSum
from macropodus.summarize.graph_base.textrank_sklearn import TextrankSklearn


# textrank of gensim
trgs = TextrankGensimSum()
# textrank of word2vec
trwv = TextrankWord2vec()
# textrank of sklearn
trsk = TextrankSklearn()


class TextRankSum:
    def __init__(self):
        self.algorithm = 'textrank'

    def summarize(self, text, num=6, model_type="textrank_word2vec"):
        """
            文本摘要
        :param text:str, like "你好！大漠帝国！" 
        :param num: int, like 3
        :param model_type: str, like "textrank_sklearn"
        :return: list
        """
        if model_type=="textrank_sklearn":
            res = trsk.summarize(text, num=num)
        elif model_type=="textrank_gensim":
            res = trgs.summarize(text, num=num)
        elif model_type=="textrank_word2vec":
            res = trwv.summarize(text, num=num)
        else:
            raise RuntimeError(" model_type must be 'textrank_textrank4zh', 'text_rank_sklearn' or 'textrank_gensim' ")

        return res


class TextRankKey:
    def __init__(self):
        self.algorithm = 'keyword'

    def keyword(self, text, num=6, score_min=0.025, model_type="keywor_word2vec"):
        if model_type=="keywor_word2vec":
            res = trwv.keyword(text, num=num, score_min=score_min)
        else:
            raise RuntimeError(" model_type must be 'keywor_word2vec'")

        return res



if __name__ == '__main__':

    doc = "和投票目标的等级来决定新的等级.简单的说。" \
           "是上世纪90年代末提出的一种计算网页权重的算法!" \
           "当时，互联网技术突飞猛进，各种网页网站爆炸式增长。" \
           "业界急需一种相对比较准确的网页重要性计算方法。" \
           "是人们能够从海量互联网世界中找出自己需要的信息。" \
           "百度百科如是介绍他的思想:PageRank通过网络浩瀚的超链接关系来确定一个页面的等级。" \
           "Google把从A页面到B页面的链接解释为A页面给B页面投票。" \
           "Google根据投票来源甚至来源的来源，即链接到A页面的页面。" \
           "一个高等级的页面可以使其他低等级页面的等级提升。" \
           "具体说来就是，PageRank有两个基本思想，也可以说是假设。" \
           "即数量假设：一个网页被越多的其他页面链接，就越重）。" \
           "质量假设：一个网页越是被高质量的网页链接，就越重要。" \
           "总的来说就是一句话，从全局角度考虑，获取重要的信。"

    text = doc.encode('utf-8').decode('utf-8')

    tr = TextRankSum()
    kw = TextRankKey()
    score_ques = tr.summarize(text, num=100, model_type="textrank_gensim") # "text_rank_sklearn")
    for sq in score_ques:
        print(sq)

    score_ques = kw.keyword(text, num=100, model_type="keywor_word2vec") # "text_rank_sklearn")
    for sq in score_ques:
        print(sq)
