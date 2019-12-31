# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/19 9:54
# @author  : Mo
# @function: cut sentences of forward of maxlength


from macropodus.base.seg_basic import SegBasic


class SegForward(SegBasic):
    def __init__(self):
        super().__init__()

    def cut(self, sentence, len_max=7):
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

if __name__ == '__main__':
    sf = SegForward()
    sentence = "macropodus是啥子呢"
    sentence = "方程的解除了零以外还有…"
    print(list(sf.cut(sentence)))

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
            # print(sen)
            count += 1
            res = sf.cut(sen)
            # print(list(res))
    time_end = time.time()
    print(time_end - time_start)
    print(count/(time_end - time_start))

    # 10000/0.17*50 = 2831272(line/s)


