# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/12/2 20:33
# @author   :Mo
# @function :topic model of NMF


from macropodus.preprocess.tools_ml import extract_chinese, tfidf_fit
from macropodus.data.words_common.stop_words import stop_words
from macropodus.preprocess.tools_ml import macropodus_cut
from macropodus.preprocess.tools_ml import cut_sentence
# sklearn
from sklearn.decomposition import NMF
import numpy as np


class NMFSum:
    def __init__(self):
        self.stop_words = stop_words.values()
        self.algorithm = 'lsi'

    def summarize(self, text, num=320, topic_min=5, judge_topic="all"):
        """

        :param text: text or list, input docs
        :param num: int, number or amount of return
        :param topic_min: int, topic number
        :param judge_topic: str, calculate ways of topic
        :return: 
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
        len_sentences_cut = len(sentences_cut)
        # 去除停用词等
        self.sentences_cut = [list(filter(lambda x: x not in self.stop_words, sc)) for sc in sentences_cut]
        self.sentences_cut = [" ".join(sc) for sc in self.sentences_cut]
        # 计算每个句子的tfidf
        sen_tfidf = tfidf_fit(self.sentences_cut)
        # 主题数, 经验判断
        topic_num = min(topic_min, int(len(sentences_cut) / 2))  # 设定最小主题数为3
        nmf_tfidf = NMF(n_components=topic_num, max_iter=320)
        res_nmf_w = nmf_tfidf.fit_transform(sen_tfidf.T) # 基矩阵 or 权重矩阵
        res_nmf_h = nmf_tfidf.components_                # 系数矩阵 or 降维矩阵

        if judge_topic:
            ### 方案一, 获取最大那个主题的k个句子
            ##################################################################################
            topic_t_score = np.sum(res_nmf_h, axis=-1)
            # 对每列(一个句子topic_num个主题),得分进行排序,0为最大
            res_nmf_h_soft = res_nmf_h.argsort(axis=0)[-topic_num:][::-1]
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
            res_nmf_h_soft_argmax = res_nmf_h[topic_t_tc_argmax].tolist()
            res_combine = {}
            for l in range(len_sentences_cut):
                res_combine[self.sentences[l]] = res_nmf_h_soft_argmax[l]
            score_sen = [(rc[1], rc[0]) for rc in sorted(res_combine.items(), key=lambda d: d[1], reverse=True)]
            #####################################################################################
        else:
            ### 方案二, 获取最大主题概率的句子, 不分主题
            res_combine = {}
            for i in range(len_sentences_cut):
                res_row_i = res_nmf_h[:, i]
                res_row_i_argmax = np.argmax(res_row_i)
                res_combine[self.sentences[i]] = res_row_i[res_row_i_argmax]
            score_sen = [(rc[1], rc[0]) for rc in sorted(res_combine.items(), key=lambda d: d[1], reverse=True)]
        num_min = min(num, int(len_sentences_cut * 0.6))
        return score_sen[0:num_min]


if __name__ == '__main__':
    nmf = NMFSum()
    doc = "多知网5月26日消息，今日，方直科技发公告，拟用自有资金人民币1.2亿元，" \
          "与深圳嘉道谷投资管理有限公司、深圳嘉道功程股权投资基金（有限合伙）共同发起设立嘉道方直教育产业投资基金（暂定名）。" \
          "该基金认缴出资总规模为人民币3.01亿元。" \
          "基金的出资方式具体如下：出资进度方面，基金合伙人的出资应于基金成立之日起四年内分四期缴足，每期缴付7525万元；" \
          "各基金合伙人每期按其出资比例缴付。合伙期限为11年，投资目标为教育领域初创期或成长期企业。" \
          "截止公告披露日，深圳嘉道谷投资管理有限公司股权结构如下:截止公告披露日，深圳嘉道功程股权投资基金产权结构如下:" \
          "公告还披露，方直科技将探索在中小学教育、在线教育、非学历教育、学前教育、留学咨询等教育行业其他分支领域的投资。" \
          "方直科技2016年营业收入9691万元，营业利润1432万元，归属于普通股股东的净利润1847万元。（多知网 黎珊）}}"

    doc = "和投票目标的等级来决定新的等级.简单的说。" \
           "是上世纪90年代末提出的一种计算网页权重的算法! " \
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

    doc = '早年林志颖带kimi上《爸爸去哪儿》的时候，当时遮遮掩掩的林志颖老婆低调探班，总让人觉得格外神秘，大概是特别不喜欢' \
           '在公众面前曝光自己日常的那种人。可能这么些年过去，心态不断调整过了，至少在微博上，陈若仪越来越放得开，晒自己带' \
           '娃照顾双子星的点滴，也晒日常自己的护肤心得，时不时安利一些小东西。都快晚上十点半，睡美容觉的最佳时候，结果才带' \
           '完一天娃的陈若仪还是不忘先保养自己，敷起了面膜。泡完澡，这次用的是一个稍微平价的面膜，脸上、甚至仔细到脖子上都' \
           '抹上了。陈若仪也是多此一举，特别说自己不是裸体，是裹着浴巾的，谁在意这个呀，目光完全被你那又长又扑闪的睫毛给吸' \
           '引住了。这也太吓人吧，怎么能够长那么长那么密那么翘。嫉妒地说一句，真的很像种的假睫毛呐。陈若仪的睫毛应该是天生' \
           '的基础好吧，要不然也不会遗传给小孩，一家子都是睫毛精，几个儿子现在这么小都是长睫毛。只是陈若仪现在这个完美状态，' \
           '一定是后天再经过悉心的呵护培养。网友已经迫不及待让她教教怎么弄睫毛了，陈若仪也是答应地好好的。各种私人物品主动' \
           '揭秘，安利一些品牌给大家，虽然一再强调是自己的日常小物，还是很让人怀疑，陈若仪是不是在做微商当网红呐，网友建议' \
           '她开个店，看这回复，也是很有意愿了。她应该不缺这个钱才对。隔三差五介绍下自己用的小刷子之类，陈若仪乐于向大家传' \
           '授自己的保养呵护之道。她是很容易就被晒出斑的肤质，去海岛参加婚礼，都要必备这几款超爱用的防晒隔离。日常用的、太' \
           '阳大时候用的，好几个种类，活得相当精致。你们按照自己的需要了解一下。画眉毛，最爱用的是intergrate的眉笔。也是个' \
           '念旧的人，除了Dior，陈若仪的另一个眉粉其中一个是她高中就开始用的Kate。一般都是大学才开始化妆修饰自己，感受得到' \
           '陈若仪从小就很爱美。各种小零小碎的化妆品，已经买过七八次的粉红胡椒抛光美体油，每天洗完澡陈若仪都会喷在肚子、大' \
           '腿、屁股和膝盖手肘，说是能保持肌肤的平滑紧致程度。每安利一样东西，总有网友要在下面问其他问题咋个办，真是相当信' \
           '任陈若仪了。每次她也很耐心的解答，"去黑头我用的是SUQQU洁面去角质按摩膏磨砂洁面洗面奶，"一定要先按摩再用。她自己' \
           '已经回购过好几次，意思是你们再了解一下。了解归了解，买不买随意。毕竟像她另一个爱用的达尔肤面膜，效果好是好，价' \
           '格据说比sk2都还要贵，不是大多数人日常能够消费得起的，大家就看个热闹就好了，还是多买多试多用才能找到最适合自己的' \
           '护肤方法。'

    sum = nmf.summarize(doc, num=320)
    for i in sum:
        print(i)


