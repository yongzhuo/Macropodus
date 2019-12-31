# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/11/31 21:33
# @author   :Mo
# @function :topic model of LDA
# @paper    :Latent Dirichlet Allocation


from macropodus.preprocess.tools_ml import extract_chinese, tfidf_fit
from macropodus.data.words_common.stop_words import stop_words
from macropodus.preprocess.tools_ml import macropodus_cut
from macropodus.preprocess.tools_ml import cut_sentence
# sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


class LDASum:
    def __init__(self):
        self.stop_words = stop_words.values()
        self.algorithm = 'lda'

    def summarize(self, text, num=8, topic_min=6, judge_topic=None):
        """
            LDA
        :param text: str
        :param num: int
        :param topic_min: int 
        :param judge_topic: boolean
        :return: 
        """
        # 切句
        if type(text) == str:
            self.sentences = cut_sentence(text)
        elif type(text) == list:
            self.sentences = text
        else:
            raise RuntimeError("text type must be list or str")
        len_sentences_cut = len(self.sentences)
        # 切词
        sentences_cut = [[word for word in macropodus_cut(extract_chinese(sentence))
                          if word.strip()] for sentence in self.sentences]
        # 去除停用词等
        self.sentences_cut = [list(filter(lambda x: x not in self.stop_words, sc)) for sc in sentences_cut]
        self.sentences_cut = [" ".join(sc) for sc in self.sentences_cut]
        # # 计算每个句子的tf
        # vector_c = CountVectorizer(ngram_range=(1, 2), stop_words=self.stop_words)
        # tf_ngram = vector_c.fit_transform(self.sentences_cut)
        # 计算每个句子的tfidf
        tf_ngram = tfidf_fit(self.sentences_cut)
        # 主题数, 经验判断
        topic_num = min(topic_min, int(len(sentences_cut) / 2))  # 设定最小主题数为3
        lda = LatentDirichletAllocation(n_topics=topic_num, max_iter=32,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=2019)
        res_lda_u = lda.fit_transform(tf_ngram.T)
        res_lda_v = lda.components_

        if judge_topic:
            ### 方案一, 获取最大那个主题的k个句子
            ##################################################################################
            topic_t_score = np.sum(res_lda_v, axis=-1)
            # 对每列(一个句子topic_num个主题),得分进行排序,0为最大
            res_nmf_h_soft = res_lda_v.argsort(axis=0)[-topic_num:][::-1]
            # 统计为最大每个主题的句子个数
            exist = (res_nmf_h_soft <= 0) * 1.0
            factor = np.ones(res_nmf_h_soft.shape[1])
            topic_t_count = np.dot(exist, factor)
            # 标准化
            topic_t_count /= np.sum(topic_t_count, axis=-1)
            topic_t_score /= np.sum(topic_t_score, axis=-1)
            # 主题最大个数占比, 与主题总得分占比选择最大的主题
            topic_t_tc = topic_t_count + topic_t_score
            topic_t_tc_argmax = np.argmax(topic_t_tc)
            # 最后得分选择该最大主题的
            res_nmf_h_soft_argmax = res_lda_v[topic_t_tc_argmax].tolist()
            res_combine = {}
            for l in range(len_sentences_cut):
                res_combine[self.sentences[l]] = res_nmf_h_soft_argmax[l]
            score_sen = [(rc[1], rc[0]) for rc in sorted(res_combine.items(), key=lambda d: d[1], reverse=True)]
            #####################################################################################
        else:
            ### 方案二, 获取最大主题概率的句子, 不分主题
            res_combine = {}
            for i in range(len_sentences_cut):
                res_row_i = res_lda_v[:, i]
                res_row_i_argmax = np.argmax(res_row_i)
                res_combine[self.sentences[i]] = res_row_i[res_row_i_argmax]
            score_sen = [(rc[1], rc[0]) for rc in sorted(res_combine.items(), key=lambda d: d[1], reverse=True)]
        num_min = min(num, int(len_sentences_cut * 0.6))
        return score_sen[0:num_min]


if __name__ == '__main__':
    lda = LDASum()
    doc = "多知网5月26日消息，今日，方直科技发公告，拟用自有资金人民币1.2亿元，" \
          "与深圳嘉道谷投资管理有限公司、深圳嘉道功程股权投资基金（有限合伙）共同发起设立嘉道方直教育产业投资基金（暂定名）。" \
          "该基金认缴出资总规模为人民币3.01亿元。" \
          "基金的出资方式具体如下：出资进度方面，基金合伙人的出资应于基金成立之日起四年内分四期缴足，每期缴付7525万元；" \
          "各基金合伙人每期按其出资比例缴付。合伙期限为11年，投资目标为教育领域初创期或成长期企业。" \
          "截止公告披露日，深圳嘉道谷投资管理有限公司股权结构如下:截止公告披露日，深圳嘉道功程股权投资基金产权结构如下:" \
          "公告还披露，方直科技将探索在中小学教育、在线教育、非学历教育、学前教育、留学咨询等教育行业其他分支领域的投资。" \
          "方直科技2016年营业收入9691万元，营业利润1432万元，归属于普通股股东的净利润1847万元。（多知网 黎珊）}}"

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

    sum = lda.summarize(doc, num=8)
    for i in sum:
        print(i)
