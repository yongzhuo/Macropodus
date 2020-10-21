# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/19 16:44
# @author  : Mo
# @function: chinese word discovery


from macropodus.data.words_common.stop_words import stop_words
from macropodus.preprocess.tools_ml import cut_sentence
from macropodus.preprocess.tools_ml import get_ngrams
from collections import Counter, OrderedDict
from functools import reduce
from operator import mul
import math
import os


class WordDiscovery:
    def __init__(self):
        from macropodus.segment import segs
        self.dict_words_freq = segs.dict_words_freq
        self.algorithm = "new-word-discovery"
        self.stop_words = stop_words
        self.total_words_len = {}
        self.total_words = 0
        self.freq_min = 3
        self.len_max = 7
        self.round = 6
        self.eps = 1e-9
        self.empty_words = [sw for sw in stop_words.values() if len(sw)==1] # 虚词

    def count_word(self, text, use_type="text"):
        """
            词频统计(句子/段落/文章)
        :param text: str, path or doc, like "大漠帝国。" or "/home/data/doc.txt"
        :param use_type: str,  "text" or "file", file of "utf-8" of "txt"
        :return: class<Counter>, word-freq
        """
        import macropodus
        self.words_count = Counter()
        if use_type=="text": # 输入为文本形式
            text = macropodus.han2zh(text)
            texts = cut_sentence(use_type=self.algorithm,
                                 text=text)  # 切句子, 如中英文的逗号/句号/感叹号
            for text in texts:
                n_grams = get_ngrams(use_type=self.algorithm,
                                     len_max=self.len_max,
                                     text=text) # 获取一个句子的所有n-gram
                self.words_count.update(n_grams)
        elif use_type=="file": # 输入为文件形式
            if not os.path.exists(text):
                raise RuntimeError("path of text must exist!")
            fr8 = open(text, "r", encoding="utf-8")
            for text in fr8:
                if text.strip():
                    text = macropodus.han2zh(text)
                    texts = cut_sentence(use_type=self.algorithm,
                                         text=text) # 切句子, 如中英文的逗号/句号/感叹号
                    for text in texts:
                        n_grams = get_ngrams(use_type=self.algorithm,
                                             len_max=self.len_max,
                                             text=text)  # 获取一个句子的所有n-gram
                        self.words_count.update(n_grams)
            fr8.close()
        else:
            raise RuntimeError("use_type must be 'text' or 'file'")
        self.total_words = sum(self.words_count.values())

    def calculate_entropy(self, boundary_type="left"):
        """
            计算左熵和右熵
        :param boundary_type: str, like "left" or "right"
        :return: None
        """
        # 获取成词的最左边和最右边的一个字
        one_collect = {}
        self.total_words_len = {}
        for k, v in self.words_count.items():
            len_k = len(k)
            if len_k >= 2:  # 词长度大于3
                if boundary_type == "right":
                    k_boundary = k[:-1]
                else:
                    k_boundary = k[1:]
                # 左右边, 保存为dict, 左右丰度
                if k_boundary in self.words_count:
                    if k_boundary not in one_collect:
                        one_collect[k_boundary] = [v]
                    else:
                        one_collect[k_boundary] = one_collect[k_boundary] + [v]
            # 计算n-gram的长度
            if len_k not in self.total_words_len:
                self.total_words_len[len_k] = [v]
            else:
                self.total_words_len[len_k] += [v]
        self.total_words_len = dict([(k, sum(v)) for k,v in self.total_words_len.items()])

        # 计算左右熵
        for k, v in self.words_select.items():
            # 从字典获取
            boundary_v = one_collect.get(k, None)
            # 计算候选词的左右凝固度, 取最小的那个
            if boundary_v:
                # 求和
                sum_boundary = sum(boundary_v)
                # 计算信息熵
                entroy_boundary = sum([-(enum_bo / sum_boundary) * math.log(enum_bo / sum_boundary, 2)
                                       for enum_bo in boundary_v])
            else:
                entroy_boundary = 0.0
            # 惩罚虚词开头或者结尾
            if (k[0] in self.empty_words or k[-1] in self.empty_words):
                entroy_boundary = entroy_boundary / len(k)
            if boundary_type == "right":
                self.right_entropy[k] = round(entroy_boundary, self.round)
            else:
                self.left_entropy[k] = round(entroy_boundary, self.round)

    def compute_entropys(self):
        """
            计算左右熵
        :param words_count:dict, like {"我":32, "你们":12} 
        :param len_max: int, like 6
        :param freq_min: int, like 32
        :return: dict
        """
        # 提取大于最大频率的词语, 以及长度在3-len_max的词语
        self.words_select = {word: count for word, count in self.words_count.items()
                             if count >= self.freq_min and " " not in word
                             and 1 < len(word) <= self.len_max
                             }
        # 计算凝固度, 左右两边
        self.right_entropy = {}
        self.left_entropy = {}
        self.calculate_entropy(boundary_type="left")
        self.calculate_entropy(boundary_type="right")

    def compute_aggregation(self):
        """
            计算凝固度
        :return: None
        """
        twl_1 = self.total_words_len[1] # ngram=1 的所有词频
        self.aggregation = {}
        for word, value in self.words_select.items():
            len_word = len(word)
            twl_n = self.total_words_len[len_word] # ngram=n 的所有词频
            words_freq = [self.words_count.get(wd, 1) for wd in word]
            probability_word = value / twl_n
            probability_chars = reduce(mul,([wf for wf in words_freq])) / (twl_1**(len(word)))
            pmi = math.log(probability_word / probability_chars, 2)
            # AMI=PMI/length_word. 惩罚虚词(避免"的", "得", "了"开头结尾的情况)
            word_aggregation = pmi/(len_word**len_word) if (word[0] in self.empty_words or word[-1] in self.empty_words) \
                                                        else pmi/len_word # pmi / len_word / len_word
            self.aggregation[word] = round(word_aggregation, self.round)

    def compute_score(self, word, value, a, r, l, rl, lambda_0, lambda_3):
        """
            计算最终得分
        :param word: str, word with prepare
        :param value: float, word freq
        :param a: float, aggregation of word
        :param r: float, right entropy of word
        :param l: float, left entropy of word
        :param rl: float, right_entropy * left_entropy
        :param lambda_0: lambda 0
        :param lambda_3: lambda 3
        :return: 
        """
        self.new_words[word] = {}
        # math.log10(self.aggregation[word]) - math.log10(self.total_words)
        self.new_words[word]["a"] = a
        self.new_words[word]["r"] = r
        self.new_words[word]["l"] = l
        self.new_words[word]["f"] = value
        # word-liberalization
        m1 = lambda_0(r)
        m2 = lambda_0(l)
        m3 = lambda_0(a)
        score_ns = lambda_0((lambda_3(m1, m2) + lambda_3(m1, m3) + lambda_3(m2, m3)) / 3)
        self.new_words[word]["ns"] = round(score_ns, self.round)
        # 乘以词频word-freq, 连乘是为了防止出现较小项
        score_s = value * a * rl * score_ns
        self.new_words[word]["s"] = round(score_s, self.round)

    def find_word(self, text, use_type="text", freq_min=2, len_max=5, entropy_min=2.0, aggregation_min=3.2,
                        use_output=True, use_avg=False, use_filter=False):
        """
            新词发现与策略
        :param text: str, path or doc, like "大漠帝国。" or "/home/data/doc.txt"
        :param use_type: str,  输入格式, 即文件输入还是文本输入, "text" or "file", file of "utf-8" of "txt"
        :param use_output: bool,  输出模式, 即最后结果是否全部输出
        :param use_filter: bool,  新词过滤, 即是否过滤macropodus词典和停用词
        :param freq_min: int, 最小词频, 大于1
        :param len_max: int, 最大成词长度, 一般为5, 6, 7
        :param entropy_min: int, 左右熵阈值, 低于则过滤
        :param aggregation_min: int, PMI(凝固度)-阈值, 低于则过滤
        :return: 
        """
        self.aggregation_min = aggregation_min
        self.entropy_min = entropy_min
        self.freq_min = freq_min
        self.len_max = len_max
        self.count_word(text=text, use_type=use_type)
        self.compute_entropys()
        self.compute_aggregation()
        self.new_words = {}
        lambda_3 = lambda m1, m2: math.log((m1 * math.e ** m2 + m2 * math.e ** m1 + self.eps) / (abs(m1 - m2) + 1), 10)
        lambda_0 = lambda x: -self.eps * x + self.eps if x <= 0 else x
        # 输出
        for word, value in self.words_select.items():
            # 过滤通用词
            if use_filter and word in self.dict_words_freq:
                continue
            # 过滤停用词
            if word in self.stop_words:
                continue
            # {"aggregation":"a", "right_entropy":"r", "left_entropy":"l", "frequency":"f",
            #  "word-liberalization":"ns", "score":"s"}
            a = self.aggregation[word]
            r = self.right_entropy[word]
            l = self.left_entropy[word]
            rl = (r+l) / 2 if use_avg else r * l
            if use_output or (use_avg and a > self.aggregation_min and rl > self.entropy_min) or \
                             (not use_avg and a > self.aggregation_min and r > self.entropy_min and l > self.entropy_min):
                self.compute_score(word, value, a, r, l, rl, lambda_0, lambda_3)

        # 排序
        self.new_words = sorted(self.new_words.items(), key=lambda x:x[1]["s"], reverse=True)
        self.new_words = OrderedDict(self.new_words)
        return self.new_words


if __name__ == '__main__':

    from macropodus.preprocess.tools_common import save_json, load_json, txt_write, txt_read

    summary = "四川发文取缔全部不合规p2p。字节跳动与今日头条。成都日报，成都市，李太白与杜甫" \
              "PageRank算法简介。" \
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

    # 新词发现-文本
    wd = WordDiscovery()
    res = wd.find_word(text=summary, use_type="text", use_avg=False, use_filter=False, use_output=True,
                       freq_min=2, len_max=5, entropy_min=2.0, aggregation_min=3.2)
    for k, v in res.items():
        print(k, v)
    print("\n#################\n")

    while True:
        print("请输入:")
        ques = input()
        res = wd.find_word(text=ques, use_type="text", use_avg=False, use_filter=False, use_output=True,
                           freq_min=2, len_max=5, entropy_min=2.0, aggregation_min=3.2)
        for k, v in res.items():
            print(k, v)
    # ms = 0
