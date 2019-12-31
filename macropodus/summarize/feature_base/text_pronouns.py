# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/8/25 20:51
# @author   :Mo
# @paper    :Sentence Extraction Based Single Document Summarization(2005)
# @function :text summary of feature-base
# @evaluate :bad, it is for english, and that's not clearly explain of formula


from macropodus.preprocess.tools_ml import macropodus_cut, jieba_tag_cut
from macropodus.data.words_common.stop_words import stop_words
from macropodus.preprocess.tools_ml import extract_chinese
from macropodus.preprocess.tools_ml import cut_sentence
from macropodus.preprocess.tools_ml import get_ngrams
# import jieba.analyse as analyse
from collections import Counter


# # jieba预训练好的idf值
# default_tfidf = analyse.default_tfidf
# # 引入TF-IDF关键词抽取接口
# tfidf = analyse.extract_tags
# # 引入TextRank关键词抽取接口
# textrank = analyse.textrank


CHAR_PUMCTUATION = ',.:;?!`\'"[]{}<>。？！，、；：“” ‘’「」『』《》（）[]〔〕【】——……—-～·《》〈〉﹏﹏.___'
CHAR_ENGLISH = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR_NUMBER = "0123456789零一二两三四五六七八九"
CHAR_CHINESE = "\u4e00-\u9fa5"
ES_MIN = 1e-9


class TextPronounsSum:
    def __init__(self):
        self.algorithm = 'text_pronouns'
        self.stop_words = stop_words.values()
        self.len_ideal = 18 # 中心句子长度, 默认

    def score_position(self):
        """
            文本长度得分
        :param sentence: 
        :return: 
        """
        score_position = []
        for i, _ in enumerate(self.sentences):
            score_standard = i / (len(self.sentences))
            if score_standard >= 0 and score_standard <= 0.1:
                score_position.append(0.17)
            elif score_standard > 0.1 and score_standard <= 0.2:
                score_position.append(0.23)
            elif score_standard > 0.2 and score_standard <= 0.3:
                score_position.append(0.14)
            elif score_standard > 0.3 and score_standard <= 0.4:
                score_position.append(0.08)
            elif score_standard > 0.4 and score_standard <= 0.5:
                score_position.append(0.05)
            elif score_standard > 0.5 and score_standard <= 0.6:
                score_position.append(0.04)
            elif score_standard > 0.6 and score_standard <= 0.7:
                score_position.append(0.06)
            elif score_standard > 0.7 and score_standard <= 0.8:
                score_position.append(0.04)
            elif score_standard > 0.8 and score_standard <= 0.9:
                score_position.append(0.04)
            elif score_standard > 0.9 and score_standard <= 1.0:
                score_position.append(0.15)
            else:
                score_position.append(0)
        return score_position

    def score_length(self):
        """
            文本长度得分
        :param sentence: 
        :return: 
        """
        score_length = []
        for i, sentence in enumerate(self.sentences):
            score_len = 1 - abs(self.len_ideal - len(sentence)) / self.len_ideal
            score_length.append(score_len)
        return score_length

    def score_tag(self):
        """
            词性打分名词-动词-代词(n,v,r)
        :return: 
        """
        score_tag = []
        for i, sen_tag_score in enumerate(self.sentences_tag_cut):
            sen_tag = sen_tag_score.values()
            tag_dict = dict(Counter(sen_tag))
            tag_n = tag_dict.get('n', 0) + tag_dict.get('nr', 0) + tag_dict.get('ns', 0) + \
                    tag_dict.get('nt', 0) + tag_dict.get('nz', 0) + tag_dict.get('ng', 0)
            tag_v = tag_dict.get('v', 0) + tag_dict.get('vd', 0) + tag_dict.get('vn', 0) + tag_dict.get('vg', 0)
            tag_p = tag_dict.get('r', 0)
            score_sen_tag = (1.2 * tag_n + 1.0 * tag_v + 0.8 * tag_p)/(len(sen_tag_score) + 1)
            score_tag.append(score_sen_tag)
        return score_tag

    def score_title(self, words):
        """
            与标题重合部分词语
        :param words: 
        :return: 
        """
        mix_word = [word for word in words if word in self.title]
        len_mix_word = len(mix_word)
        len_title_word = len(self.title)
        return (len_mix_word + 1.0) / (len_mix_word + 2.0) / len_title_word

    def summarize(self, text, num=320, title=None):
        """
            文本句子排序
        :param docs: list
        :return: list
        """
        # 切句
        if type(text) == str:
            self.sentences = cut_sentence(text)
        elif type(text) == list:
            self.sentences = text
        else:
            raise RuntimeError("text type must be list or str")
        self.title = title
        if self.title:
            self.title = macropodus_cut(title)
        # 切词,含词性标注
        self.sentences_tag_cut = [jieba_tag_cut(extract_chinese(sentence)) for sentence in self.sentences]
        # 词语,不含词性标注
        sentences_cut = [[jc for jc in jtc.keys() ] for jtc in self.sentences_tag_cut]
        # 去除停用词等
        self.sentences_cut = [list(filter(lambda x: x not in self.stop_words, sc)) for sc in sentences_cut]
        # 词频统计
        self.words = []
        for sen in self.sentences_cut:
            self.words = self.words + sen
        self.word_count = dict(Counter(self.words))
        # 按频次计算词语的得分, 得到self.word_freq=[{'word':, 'freq':, 'score':}]
        self.word_freqs = {}
        self.len_words = len(self.words)
        for k, v in self.word_count.items():
            self.word_freqs[k] = v * 0.5 / self.len_words
        # uni_bi_tri_gram特征
        [gram_uni, gram_bi, gram_tri] = get_ngrams("".join(self.sentences), ns=[1, 2, 3])
        ngrams = gram_uni + gram_bi + gram_tri
        self.ngrams_count = dict(Counter(ngrams))
        # 句子位置打分
        scores_posi = self.score_position()
        # 句子长度打分
        scores_length = self.score_length()
        # 句子词性打分, 名词(1.2)-代词(0.8)-动词(1.0)
        scores_tag = self.score_tag()

        res_rank = {}
        self.res_score = []
        for i in range(len(sentences_cut)):
            sen_cut = self.sentences_cut[i]  # 句子中的词语
            # ngram得分
            [gram_uni_, gram_bi_, gram_tri_] = get_ngrams(self.sentences[i], ns=[1, 2, 3]) # gram_uni_bi_tri(self.sentences[i])
            n_gram_s = gram_uni_ + gram_bi_ + gram_tri_
            score_ngram = sum([self.ngrams_count[ngs] if ngs in self.ngrams_count else 0 for ngs in n_gram_s]) / (len(n_gram_s) + 1)
            # 句子中词语的平均长度
            score_word_length_avg = sum([len(sc) for sc in sen_cut])/(len(sen_cut)+1)
            score_posi = scores_posi[i]
            score_length = scores_length[i]
            score_tag = scores_tag[i]
            if self.title:  # 有标题的文本打分合并
                score_title = self.score_title(sen_cut)
                score_total = (score_title * 0.5 + score_ngram * 2.0 + score_word_length_avg * 0.5 +
                               score_length * 0.5 + score_posi * 1.0 + score_tag * 0.6) / 6.0
                # 可查阅各部分得分统计
                self.res_score.append(["score_title", "score_ngram", "score_word_length_avg",
                                       "score_length", "score_posi", "score_tag"])
                self.res_score.append([score_title, score_ngram, score_word_length_avg,
                                       score_length, score_posi, score_tag, self.sentences[i]])
            else:  # 无标题的文本打分合并
                score_total = (score_ngram * 2.0 + score_word_length_avg * 0.5 + score_length * 0.5 +
                               score_posi * 1.0 + score_tag * 0.6) / 5.0
                # 可查阅各部分得分统计
                self.res_score.append(["score_ngram", "score_word_length_avg",
                                       "score_length", "score_posi", "score_tag"])
                self.res_score.append([score_ngram, score_word_length_avg,
                                       score_length, score_posi, score_tag, self.sentences[i]])
            res_rank[self.sentences[i].strip()] = score_total
        # 最小句子数
        num_min = min(num, int(len(self.word_count) * 0.6))
        res_rank_sort = sorted(res_rank.items(), key=lambda rr: rr[1], reverse=True)
        res_rank_sort_reverse = [(rrs[1], rrs[0]) for rrs in res_rank_sort][0:num_min]
        return res_rank_sort_reverse



if __name__ == '__main__':
    sen = "自然语言理解（NLU，Natural Language Understanding）: 使计算机理解自然语言（人类语言文字）等，重在理解。"
    tp = TextPronounsSum()
    docs ="和投票目标的等级来决定新的等级.简单的说。" \
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

    docs1 = "/article/details/98530760。" \
           "CSDN\n。" \
           "文本生成NLG，不同于文本理解NLU（例如分词、词向量、分类、实体提取。" \
          "是重在文本生成的另一种关键技术（常用的有翻译、摘要、同义句生成等）。" \
          "传统的文本生成NLG任务主要是抽取式的，生成式的方法看起来到现在使用也没有那么普遍。" \
          "现在，我记录的是textrank，一种使用比较广泛的抽取式关键句提取算法。" \
          "版权声明：本文为CSDN博主「大漠帝国」的原创文章，遵循CC 4.0 by-sa版权协议，" \
          "转载请附上原文出处链接及本声明。原文链接：https://blog.csdn.net/rensihui" \
           "CSDN是神"
    sums = tp.summarize(docs)
    for sum_ in sums:
        print(sum_)

    # ran_20 = range(20)
    # print(type(ran_20))
    # print(ran_20)
    # idx = [1,2,3]
    # idx.pop(1)
    # print(idx)
    # print(max([1,2,3,4]))




