# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/3 20:01
# @author  : Mo
# @function: segnment of statistics


from macropodus.preprocess.tools_common import re_continue
from macropodus.base.seg_basic import SegBasic
from math import log
import re

__all__ = ["cut_dag",
           "cut_forward",
           "cut_reverse",
           "cut_bidirectional",
           "cut_search"]

re_han = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-]+)", re.U)
re_skip = re.compile("(\r\n|\s)", re.U)


class SegStatistics(SegBasic):
    def __init__(self, use_cache):
        self.algorithm = "chinese-word-segnment"
        super().__init__(use_cache)

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

    def cut_dag(self, sentence):
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

    def cut_forward(self, sentence, len_max=7):
        """
            正向最大切词
        :param sentence: str, like '大漠帝国'
        :param len_max: int, like 32
        :return: yield
        """
        len_sen = len(sentence)
        i = 0
        while i < len_sen: # while判断条件
            flag = False   # flag标志位,确定有没有在字典里边的单字词或多字词
            for j in range(min(len_sen+1, i+len_max), -i, -1):  # 遍历从当前字到句子末尾可能成词的部分, 从最后i+len_max算起
                word_maybe = sentence[i:j] # 正向可能成词的语
                if word_maybe in self.dict_words_freq:  # 是否在字典里边
                    i = j  # 成词前标志i向后移动
                    flag = True  # flag标志位变化
                    yield word_maybe
                    break  # 成词则跳出循环
            if not flag: # 未选中后单个字的情况
               yield sentence[i]
               i += 1

    def cut_reverse(self, sentence, len_max=7):
        """
            反向最大切词
        :param sentence: str, like '大漠帝国'
        :param len_max: int, like 32
        :return: yield
        """
        len_sen = len(sentence)
        i = len_sen
        res = []
        while i > 0:  # while判断条件
            flag = False  # flag标志位,确定有没有在字典里边的单字词或多字词
            for j in range(max(0, i - len_max), i):  # 遍历从句子末尾向前可能成词的部分, 从最后i-len_max算起
                word_maybe = sentence[j:i]  # 正向可能成词的语
                if word_maybe in self.dict_words_freq:  # 是否在字典里边
                    i = j  # 成词前标志i向后移动
                    flag = True  # flag标志位变化
                    res.append(word_maybe)
                    # yield word_maybe
                    break  # 成词则跳出循环
            if not flag:  # 未选中后单个字的情况
                i -= 1
                # yield sentence[i]
                res.append(sentence[i])
        for i in range(len(res)-1, 0, -1):
            yield res[i]
        # return res

    def cut_bidirectional(self, sentence):
        """
            最大双向词典切词, 即最大正向切词与最大反向切词合并, 选择词数小的那个返回
        :param sentence: str
        :return: 
        """
        res_forward = self.cut_forward(sentence)
        res_reverse = self.cut_reverse(sentence)
        res_forward_list = list(res_forward)
        res_reverse_list = list(res_reverse)
        len_res_forward = len(res_forward_list)
        len_res_reverse = len(res_reverse_list)
        if len_res_forward >= len_res_reverse:
            for rrl in res_reverse_list:
                yield rrl
        else:
            for rfl in res_forward_list:
                yield rfl

    def cut_search(self, sentence):
        """
            搜索引擎切词, 全切词
        :param sentence: str, like "大漠帝国"
        :return: yield
        """
        DAG = self.build_dag(sentence)  # 根据sentence构建有向图dag
        for k, v in DAG.items():
            for vi in v:
                yield sentence[k:vi+1]  # 遍历无向图, 返回可能存在的所有切分

    def cut(self, sentence, type_cut="cut_dag"):
        """
            切词总函数
            cut_block, 代码来自jieba项目
            code from: https://github.com/fxsjy/jieba
        :param sentence:str, like '大漠帝国, macropodus, 中国斗鱼' 
        :param type_cut: str, like 'cut_dag', 'cut_forward', 'cut_reverse', 'cut_bidirectional', 'cut_search'
        :return: yield, like ['大漠帝国', ',', 'macropodus', ',', '中国斗鱼']
        """
        if type_cut=="cut_dag":
            cut_block = self.cut_dag
        elif type_cut=="cut_forward":
            cut_block = self.cut_forward
        elif type_cut=="cut_reverse":
            cut_block = self.cut_reverse
        elif type_cut=="cut_bidirectional":
            cut_block = self.cut_bidirectional
        elif type_cut=="cut_search":
            cut_block = self.cut_search
        else:
            raise RuntimeError("type_cut must be 'cut_dag', 'cut_forward', 'cut_reverse', 'cut_bidirectional', 'cut_search'")
        blocks = re_han.split(sentence)
        cut_all = False
        for block in blocks:
            if not block:
                continue
            if re_han.match(block):
                for word in cut_block(block):
                    yield word
            else:
                tmp = re_skip.split(block)
                for x in tmp:
                    if re_skip.match(x):
                        yield x
                    elif not cut_all:
                        for xx in x:
                            yield xx
                    else:
                        yield x


if __name__ == '__main__':
    sd = SegStatistics(True)
    sd.add_word(str('知识图谱'))
    sd_search = sd.cut_search("已结婚的和尚未结婚的青年都要实行计划生育")
    print(list(sd_search))
    # for i in range(50000):
    sd_enum = sd.cut_dag(sentence="已结婚的和尚未结婚的青年都要实行计划生育")
    print(list(sd_enum))
    sd_enum = sd.cut_dag(sentence='what‘syournamesirareyouok!')
    print(list(sd_enum))
    # 测试性能
    from macropodus.preprocess.tools_common import txt_read, txt_write
    from macropodus.conf.path_config import path_root
    import time
    path_wordseg_a = path_root.replace("macropodus", "") + "/test/tet/ambiguity.txt"
    sentences = txt_read(path_wordseg_a)

    time_start = time.time()
    count = 0
    for i in range(50000):
        for sen in sentences:
            count += 1
            res = sd.cut_search(sen)
            # print(list(res))
    time_end = time.time()
    print(time_end-time_start)
    print(count/(time_end - time_start))


# win10测试, i7 8th + 16G RAM
# 10000/0.17*50 = 2864136(line/s)
# 50000/0.87*50 = 2872092(line/s)
