# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/8/26 23:42
# @author   :Mo
# @function :text summarize of extraction of word significance
# @paper    :The Automatic Creation of Literature Abstracts*
# @url      :http://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf


from macropodus.data.words_common.stop_words import stop_words
from macropodus.preprocess.tools_ml import extract_chinese
from macropodus.preprocess.tools_ml import macropodus_cut
from macropodus.preprocess.tools_ml import cut_sentence
from collections import Counter


class WordSignificanceSum:
    def __init__(self):
        """
        features:
            1. words mix in title and sentence
            2. keywords in sentence
            3. Position of sentence
            4. Length of sentence
        """
        self.algorithm = 'word_significance'
        self.stop_words = stop_words.values()
        self.num = 0

    def summarize(self, text, num=320):
        """
            根据词语意义确定中心句
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
        # 词频统计
        self.words = []
        for sen in self.sentences_cut:
            self.words = self.words + sen
        self.word_count = dict(Counter(self.words))
        self.word_count_rank = sorted(self.word_count.items(), key=lambda f: f[1], reverse=True)
        # 最小句子数
        num_min = min(num, int(len(self.word_count)*0.6))
        # 词语排序, 按照词频
        self.word_rank = [wcr[0] for wcr in self.word_count_rank][0:num_min]
        res_sentence = []
        # 抽取句子, 顺序, 如果词频高的词语在句子里, 则抽取
        for word in self.word_rank:
            for i in range(0, len(self.sentences)):
                # 当返回关键句子到达一定量, 则结束返回
                if len(res_sentence) < num_min:
                    added = False
                    for sent in res_sentence:
                        if sent == self.sentences[i]: added = True
                    if (added == False and word in self.sentences[i]):
                        res_sentence.append(self.sentences[i])
                        break
        # 只是计算各得分,没什么用
        len_sentence = len(self.sentences)
        res_sentence = [(1-1/(len_sentence+len_sentence/(k+1)), rs) for k, rs in enumerate(res_sentence)]
        return res_sentence


if __name__ == "__main__":
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

    doc1 = "多知网. "\
          "多知网5月26日消息，今日，方直科技发公告，拟用自有资金人民币1.2亿元，" \
          "与深圳嘉道谷投资管理有限公司、深圳嘉道功程股权投资基金（有限合伙）共同发起设立嘉道方直教育产业投资基金（暂定名）。" \
          "该基金认缴出资总规模为人民币3.01亿元。" \
          "基金的出资方式具体如下：出资进度方面，基金合伙人的出资应于基金成立之日起四年内分四期缴足，每期缴付7525万元；" \
          "各基金合伙人每期按其出资比例缴付。合伙期限为11年，投资目标为教育领域初创期或成长期企业。" \
          "截止公告披露日，深圳嘉道谷投资管理有限公司股权结构如下:截止公告披露日，深圳嘉道功程股权投资基金产权结构如下:" \
          "公告还披露，方直科技将探索在中小学教育、在线教育、非学历教育、学前教育、留学咨询等教育行业其他分支领域的投资。" \
          "方直科技2016年营业收入9691万元，营业利润1432万元，归属于普通股股东的净利润1847万元。（多知网 黎珊）}}"

    ws = WordSignificanceSum()
    res = ws.summarize(doc, num=6)
    for r in res:
        print(r)
