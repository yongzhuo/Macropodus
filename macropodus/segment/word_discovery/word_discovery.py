# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/19 16:44
# @author  : Mo
# @function: chinese word discovery


from macropodus.preprocess.tools_ml import cut_sentence
from macropodus.preprocess.tools_ml import get_ngrams
from collections import Counter
import math
import os


class WordDiscovery:
    def __init__(self):
        self.algorithm = "new-word-discovery"
        self.total_words = 0
        self.freq_min = 3
        self.len_max = 7

    def count_word(self, text, use_type="text"):
        """
            词频统计(句子/段落/文章)
        :param text: str, path or doc, like "大漠帝国。" or "/home/data/doc.txt"
        :param use_type: str,  "text" or "file", file of "utf-8" of "txt"
        :return: class<Counter>, word-freq
        """
        self.words_count = Counter()
        if use_type=="text": # 输入为文本形式
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
        for k, v in self.words_count.items():
            len_k = len(k)
            if len_k >= 3:  # 词长度大于3
                if boundary_type == "right":
                    k_boundary = k[:-1]
                else:
                    k_boundary = k[1:]
                if k_boundary in self.words_select:  # 左右边, 保存为dict
                    if k_boundary not in one_collect:
                        one_collect[k_boundary] = [v]
                    else:
                        one_collect[k_boundary] = one_collect[k_boundary] + [v]

        # 计算成词的互信息
        for k, v in self.words_select.items():
            # 从字典获取
            boundary_v = one_collect.get(k, None)
            # 计算候选词的左右凝固度, 取最小的那个
            if boundary_v:
                sum_boundary = sum(boundary_v)  # 求和
                # 计算信息熵
                entroy_boundary = sum([-(enum_bo / sum_boundary) * math.log(enum_bo / sum_boundary)
                                       for enum_bo in boundary_v])
            else:
                entroy_boundary = 0.0
            if boundary_type == "right":
                self.right_entropy[k] = entroy_boundary
            else:
                self.left_entropy[k] = entroy_boundary

    def compute_entropys(self):
        """
            计算凝固度
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
        # self.words_count.clear() # 清除变量

    def compute_aggregation(self):
        """
            计算凝固度
        :return: None
        """
        self.aggregation = {}
        for word, value in self.words_select.items():
            len_word = len(word)
            score_aggs = []
            for i in range(1, len_word): # 候选词的左右两边各取一个字
                word_right = word[i:]
                word_left = word[:i]
                value_right = self.words_select.get(word_right, self.freq_min)
                value_left = self.words_select.get(word_left, self.freq_min)
                # score_agg_single = math.log(value) - math.log(value_right * value_left)
                score_agg_single = value / (value_right * value_left)
                # score_agg_single = math.log10(value) - math.log10(self.total_words) -math.log10((value_right * value_left))
                score_aggs.append(score_agg_single)
            self.aggregation[word] = min(score_aggs)

    def find_word(self, text, use_type="text", freq_min=2, len_max=7, entropy_min=1.2, aggregation_min=0.5, use_avg=False):
        """
            新词发现与策略
        :param text: str, path or doc, like "大漠帝国。" or "/home/data/doc.txt"
        :param use_type: str,  "text" or "file", file of "utf-8" of "txt"
        :param freq_min: int, 最小词频, 大于1
        :param len_max: int, 最大成词长度, 一般为5, 6, 7
        :param entropy_min: int, 最小词频, 大于1
        :param aggregation_min: int, 最大成词长度, 一般为5, 6, 7
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
        # 输出
        for word,value in self.words_select.items():
            if not use_avg and self.aggregation[word] > self.aggregation_min \
                    and self.right_entropy[word] > self.entropy_min and self.left_entropy[word] > self.entropy_min:
                self.new_words[word] = {}
                # {"aggregation":"agg", "right_entropy":"r", "left_entropy":"l", "frequency":"f", "score":"s"}
                self.new_words[word]["a"] = self.aggregation[word] # math.log10(self.aggregation[word]) - math.log10(self.total_words)
                self.new_words[word]["r"] = self.right_entropy[word]
                self.new_words[word]["l"] = self.left_entropy[word]
                self.new_words[word]["f"] = value / self.total_words
                self.new_words[word]["s"] = self.new_words[word]["f"] * self.new_words[word]["a"] * \
                                                (self.right_entropy[word] + self.left_entropy[word])
            elif use_avg and self.aggregation[word] > self.aggregation_min \
                    and (self.right_entropy[word] + self.left_entropy[word]) > 2 * self.entropy_min:
                self.new_words[word] = {}
                # {"aggregation":"agg", "right_entropy":"r", "left_entropy":"l", "frequency":"f", "score":"s"}
                self.new_words[word]["a"] = self.aggregation[word]
                self.new_words[word]["r"] = self.right_entropy[word]
                self.new_words[word]["l"] = self.left_entropy[word]
                self.new_words[word]["f"] = value / self.total_words
                self.new_words[word]["s"] = self.new_words[word]["f"] * self.new_words[word]["a"] * \
                                                (self.right_entropy[word] + self.left_entropy[word])

        return self.new_words


if __name__ == '__main__':
    text = "PageRank算法简介。" \
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
    # wc = count_word(text)
    # path = "data/poet_tangsong.csv"
    # wd = WordDiscovery()
    # res = wd.find_word(text=path, use_type="file", freq_min=2, len_max=6, entropy_min=1.2, aggregation_min=0.4)
    # from macropodus.preprocess.tools_common import txt_write
    # import json
    # res_s = json.dumps(res)
    # txt_write([res_s], "res_s.txt")
    # print(res)
    # with open("res_s.txt", "r", encoding="utf-8") as fd:
    #     ff = fd.readlines()[0]
    #     res_ = json.loads(ff)
    #     res_soft = sorted(res_.items(), key=lambda d: d[1]['score'], reverse=True)
    wd = WordDiscovery()
    res = wd.find_word(text=text, use_type="text", use_avg=True, freq_min=2, len_max=7, entropy_min=0.4, aggregation_min=1.2)
    for k, v in res.items():
        print(k, v)
    while True:
        print("请输入:")
        ques = input()
        res = wd.find_word(text=ques, use_type="text", use_avg=True, freq_min=2, len_max=7, entropy_min=0.52, aggregation_min=1.2)
        for k, v in res.items():
            print(k, v)
    # gg = 0
