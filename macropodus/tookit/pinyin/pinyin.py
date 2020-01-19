# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/1/8 21:40
# @author  : Mo
# @function: 汉字转拼音(zh2pinyin)


from macropodus.preprocess.tools_common import re_zh_cn, load_json
from macropodus.preprocess.tools_ml import macropodus_cut
from macropodus.conf.path_config import path_dict_pinyin
from collections import defaultdict


class PinYin:
    def __init__(self):
        self.algorithm = "pinyin"
        self.dict_pinyin = defaultdict()
        self.load_pinyin_dict()

    def load_pinyin_dict(self):
        """
            加载默认的拼音pinyin字典
        :return: None
        """
        dict_pinyin = load_json(path_dict_pinyin)[0] # 加载json字典文件
        for k, v in dict_pinyin.items():
            self.dict_pinyin[k] = v

    def pinyin(self, text):
        """
            中文(大陆)转拼音
        :param text: str, like "大漠帝国"
        :return: list, like ["da", "mo", "di", "guo"]
        """
        res_pinyin = []
        # 只选择中文(zh), split筛选
        text_re = re_zh_cn.split(text)
        for tr in text_re:
            if re_zh_cn.match(tr):
                # 切词
                tr_cut = macropodus_cut(tr)
                for trc in tr_cut: # 切词后的词语
                    # get words from dict of default
                    trc_pinyin = self.dict_pinyin.get(trc)
                    if trc_pinyin: res_pinyin += trc_pinyin
                    else: # 单个字的问题
                        for trc_ in trc:
                            # get trem from dict of default
                            trc_pinyin = self.dict_pinyin.get(trc_)
                            if trc_pinyin: res_pinyin += trc_pinyin
        return res_pinyin


if __name__ == "__main__":
    text = "macropodus是一种中国产的淡水鱼，广泛分布于两广地区，abcdefghijklmnopqrstuvwxyz"
    py = PinYin()
    res = py.pinyin(text)
    print(res)
    while True:
        print("请输入:")
        ques = input()
        print(py.pinyin(ques))
