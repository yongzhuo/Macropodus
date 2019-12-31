# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/19 9:58
# @author  : Mo
# @function: segmentation of maximum probability using dictionary


from macropodus.preprocess.tools_common import re_continue
from macropodus.base.seg_basic import SegBasic
from math import log


class SegDAG(SegBasic):
    def __init__(self):
        super().__init__()

    def build_dag(self, sentence, len_word_max=105):
        """
            构建句子的词典概率有向图;
            jieba使用的是前缀字典替代前缀树,内存比前缀树小,且比前缀树快;
            基本思想是构建'大漠帝国:132','大漠帝','大漠:640','大':1024等，没有则置为0,
            搜索时候前缀不存在就跳出,不用继续下去了
        :param sentence: str, like '大漠帝国是谁'
        :param sentence: int, like 132
        :return: dict, like {0:[0,1], 1:[1]}
        """
        len_sen = len(sentence)
        dag_sen = {}
        for i in range(len_sen):  # 前向遍历, 全切分
            enum_j = [i]          # 单个字就是它本身
            for j in range(i+1, min(len_sen, i+len_word_max)):    # 遍历从当前字到句子末尾可能成词的部分, 当前的不取, 设置最大成词长度为132
                word_maybe = sentence[i:j+1]
                if word_maybe in self.dict_words_freq:
                    enum_j.append(j)
            dag_sen[i] = enum_j
        return dag_sen

    def calculate_prob(self, sentence, DAG, route):
        """
            动态规划求取最大概率, 代码来自jieba项目
            code from: https://github.com/fxsjy/jieba
        :param sentence: str, input of sentence, like "大漠帝国是谁?"
        :param DAG: dict, 
        :param route: dict, 
        :return: None
        """
        len_sen = len(sentence)
        route[len_sen] = (0, 0)
        log_total = log(self.num_words)
        for index in range(len_sen - 1, -1, -1): # 动态规划
            route[index] = max((log(self.dict_words_freq.get(sentence[index:x + 1]) or 1)
                              - log_total + route[x + 1][0], x) for x in DAG[index])

    def cut(self, sentence):
        """
            seg_dag字典最大概率切词, 代码来自jieba项目
            code from: https://github.com/fxsjy/jieba
        :param sentence: str, input of sentence, like "大漠帝国是谁?"
        :return: None
        """
        len_sen = len(sentence)
        word_temp = ''
        route = {}
        i = 0
        DAG = self.build_dag(sentence) # 根据sentence构建有向图dag
        self.calculate_prob(sentence, DAG, route) # 动态规划计算概率最大的路径
        while i < len_sen:
            j = route[i][1] + 1 # 获取index, i为成词的begin, j为成词的end
            word_ch = sentence[i:j] # 概率成词
            if (j-i<2) and re_continue.match(word_ch): # 单个字判断是否为连续, 字母-数字-.-@等为连续
                word_temp += word_ch
                i = j
            else: # 成词后返回一个yield可迭代对象, yield后转list有点耗时
                if word_temp: # 有word_temp的情况下 word_ch也没有迭代返回
                    yield word_temp
                    word_temp = ''
                yield word_ch
                i = j
        if word_temp: # 最后一个成词为"字母-数字-.-@等为连续"的情况
            yield word_temp


if __name__ == '__main__':
    sd = SegDAG()
    sd.add_word(str('知识图谱'))

    # for i in range(50000):
    sd_enum = sd.cut(sentence='apple_pir大漠帝国我再也找不到了')
    print(list(sd_enum))

    # 测试性能
    from macropodus.preprocess.tools_common import txt_read, txt_write
    from macropodus.conf.path_config import path_root
    import time
    path_wordseg_a = path_root.replace("macropodus", "") + "/test/tet/ambiguity.txt"
    sentences = txt_read(path_wordseg_a)

    time_start = time.time()
    count = 0
    for i in range(10000):
        for sen in sentences:
            # print("原句:"+sen)
            count += 1
            res = sd.cut(sen)
            # print(list(res))
    time_end = time.time()
    print(time_end-time_start)
    print(count/(time_end - time_start))

while True:
    print("请输入:")
    sen = input()
    print(list(sd.cut(sen)))
# win10测试, i7 8th + 16G RAM
# 10000/0.17*50 = 2864136(line/s)
# 50000/0.87*50 = 2872092(line/s)


