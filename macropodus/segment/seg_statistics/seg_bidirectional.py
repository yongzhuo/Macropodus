# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/19 9:55
# @author  : Mo
# @function: cut sentences of forward of reverse of maxlength


from macropodus.segment.seg_statistics.seg_forward import SegForward
from macropodus.segment.seg_statistics.seg_reverse import SegReverse


class SegBidirectional(object):
    def __init__(self):
        self.seg_forward = SegForward()
        self.seg_reverse = SegReverse()

    def cut(self, sentence):
        """
            最大双向词典切词, 即最大正向切词与最大反向切词合并, 选择词数小的那个返回
        :param sentence: str
        :return: 
        """
        res_forward = self.seg_forward.cut(sentence)
        res_reverse = self.seg_reverse.cut(sentence)
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


if __name__ == '__main__':
    sb = SegBidirectional()
    sentence = "研究生命科学研究生命科学"
    print(list(sb.cut(sentence)))

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
            count += 1
            res = sb.cut(sen)
            # print(list(res))
    time_end = time.time()
    print(time_end - time_start)
    print(count/(time_end - time_start))
    # yield
    # 10000/0.17*50 = 2500*50    = 2896810(line/s)
    # 50000/0.90*50 = 2500000/20 = 2763600(line/s)