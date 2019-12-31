# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/19 9:54
# @author  : Mo
# @function: cut sentences of reverse of maxlength


from macropodus.base.seg_basic import SegBasic


class SegReverse(SegBasic):
    def __init__(self):
        super().__init__()

    def cut(self, sentence, len_max=7):
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


if __name__ == '__main__':
    a = max(0,5)
    sf = SegReverse()
    sentence = "研究生命科学\t研究 生命 科学"
    print(list(sf.cut(sentence)))
    print(list(sf.cut("")))

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
            # print(sen)
            count += 1
            res = (sf.cut(sen))
            # print(res)
    time_end = time.time()
    print(time_end-time_start)
    print(count/(time_end - time_start))

    # 10000/0.18*50 = 2500*50    = 2784226(line/s)
    # 50000/0.98*50 = 2500000/20 = 2550109(line/s)

