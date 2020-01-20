# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/1/16 22:12
# @author  : Mo
# @function: only model predict


import pathlib
import sys
import os
project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
sys.path.append(project_path)

from macropodus.preprocess.tools_ml import extract_chinese, macropodus_cut
from tensorflow.python.keras.models import model_from_json
from macropodus.preprocess.tools_common import load_json
from keras_bert import Tokenizer
import numpy as np
import macropodus
import codecs
import os


# 常规
class AlbertBilstmPredict:
    def __init__(self, path_dir, custom_objects=None):
        self.custom_objects = custom_objects
        self.path_dir = path_dir
        self.tokenizer_init()
        self.l2i_i2l_init()
        self.params_init()
        self.model_init()

    def model_init(self):
        """模型初始化"""
        path_graph = os.path.join(self.path_dir, "graph.json")
        path_model = os.path.join(self.path_dir, "model.h5")
        # 加载模型结构
        self.model = model_from_json(open(path_graph, "r", encoding="utf-8").read(),
                                     custom_objects=self.custom_objects if self.custom_objects
                                     else macropodus.custom_objects)
        # 加载模型权重
        self.model.load_weights(path_model)

    def tokenizer_init(self):
        """字典"""
        # reader tokenizer
        token2idx = {}
        path_dict = os.path.join(self.path_dir, "vocab.txt")
        with codecs.open(path_dict, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token2idx[token] = len(token2idx)
        self.tokenizer = Tokenizer(token2idx)

    def params_init(self):
        """超参数初始化"""
        # params
        path_params = os.path.join(self.path_dir, "params.json")
        self.params = load_json(path_params)
        self.len_max = self.params["len_max"]

    def l2i_i2l_init(self):
        """类别与数字项目转化"""
        # l2i_i2l
        path_l2i_i2l = os.path.join(self.path_dir, "l2i_i2l.json")
        self.l2i_i2l = load_json(path_l2i_i2l)

    def sentence2idx(self, text, second_text=None):
        """数据预处理"""
        # text = extract_chinese(str(text).upper())
        input_id, input_type_id = self.tokenizer.encode(first=text, second=second_text, max_len=self.len_max)
        input_mask = len([1 for ids in input_id if ids != 0])
        return [input_id, input_type_id, input_mask]

    def predict(self, quess):
        """
            预测多个问句, list不要太大, 最好是1-64条句子为佳。每条句子最大长度支持126
        :param quess: list, like ["桂林是个好地方", "平乐县印山亭"]
        :return: list<dict>, like [[{"entity":"桂林","start":0, "end":2, "type":"LOC"}], [{"entity":"平乐县","start":0, "end":3, "type":"LOC"}]]
        """
        quess_encode = [self.sentence2idx(ques) for ques in quess]
        x_ = np.array(quess_encode)
        x_1 = np.array([x[0] for x in x_])
        x_2 = np.array([x[1] for x in x_])
        x_3 = np.array([x[2] for x in x_])
        ress = self.model.predict([x_1, x_2, x_3])
        ress_idxs = [[np.argmax(rl) for rl in res_list] for res_list in ress.tolist()]
        ress_label = [[self.l2i_i2l["i2l"][str(ri)] if str(ri) in self.l2i_i2l["i2l"] else "O" for ri in res_idxs]
                      for res_idxs in ress_idxs]
        ress_select = [self.tagger(quess[i], ress_label[i][1:len(quess[i]) + 1]) for i in range(len(quess))]
        return ress_select

    def predict_single(self, ques):
        """
            预测单个问句
        :param ques: str, like "桂林是个好地方"
        :return: list<dict>, like [{"entity":"桂林","start":0, "end":2, "type":"LOC"}]
        """
        mode_input = self.sentence2idx(ques)
        x_ = np.array([mode_input])
        x_1 = np.array([x[0] for x in x_])
        x_2 = np.array([x[1] for x in x_])
        x_3 = np.array([x[2] for x in x_])
        res = self.model.predict([x_1, x_2, x_3])
        res_list = res.tolist()[0]
        res_idxs = [np.argmax(rl) for rl in res_list]
        res_label = [self.l2i_i2l["i2l"][str(ri)] if str(ri) in self.l2i_i2l["i2l"] else "O" for ri in res_idxs]
        res_tag = res_label[1:len(ques) + 1]
        return self.tagger(ques, res_tag)

    def tagger(self, text, tags):
        """
            将label_sequence的结果标准化
        :param text: str, like "那南宁在哪里？"
        :param tags: list, like ["CLS", "O", "B-LOC", "I-LOC", "O"]
        :return: list<dict>, like [{"entity":"南宁","start":1, "end":3, "type":"LOC"}]
        """
        tag_last = ""
        entity = ""
        start = 0
        index = 0
        res = []
        # 充分考虑实体再句前, 句中, 句末, 连续出现实体等情况
        for char, tag in zip(text, tags):
            if tag[0] == "B":
                index = text.find(char)
                if entity != "":
                    start = text.find(entity)
                    res.append(
                        {"entity": entity, "start": start,
                         "end": index, "type": tag_last[2:]})
                    entity = ""
                entity += char
                start = index
            elif tag[0] == "I":
                if entity == "":
                    pass
                else:
                    entity += char
            elif tag[0] == "O":
                if entity != "":
                    start = text.find(entity)
                    res.append({"entity": entity, "start": start,
                                "end": index, "type": tag_last[2:]})
                    entity = ""
                if len(tag)>1:
                    index = text.find(char)
                    if entity != "":
                        start = text.find(entity)
                        res.append(
                            {"entity": entity, "start": start,
                             "end": index, "type": tag_last[2:]})
                        entity = ""
                    entity += char
                    start = index
            else:
                entity = ""
                start = index
            index = index + 1
            tag_last = tag

        if entity != "":
            start = text.find(entity)
            res.append({"entity": entity, "start": start,
                        "end": index, "type": tag_last[2:]})
        return res

    def pos_tags(self, res):
        """
            词性标注专用。list不要太大, 最好是1-64条句子为佳。每条句子最大长度支持126
        :param res: str or list, like "广西省桂林市平乐县" or ["广西省桂林市", "平乐县"]
        :return: list<tuple>, like [(广西省, ), (桂林市, ), (平乐县, )]
        """
        if type(res)==str:
            res = [res]
        ress_pred = self.predict(res)
        wpts = []
        for ress in ress_pred:
            word_pos_tag = []
            for res in ress:
                word = res["entity"]
                pos_tag = res["type"]
                word_pos_tag.append((word, pos_tag))
            wpts.append(word_pos_tag)
        return wpts

    def pos_tag(self, res):
        """
            词性标注专用。list不要太大, 最好是1-64条句子为佳。每条句子最大长度支持126
        :param res: str or list, like "广西省桂林市平乐县"
        :return: list<tuple>, like [(广西省, ), (桂林市, ), (平乐县, )]
        """
        res_pred = self.predict_single(res)
        wpts = []
        word_pos_tag = []
        for res in res_pred:
            word = res["entity"]
            pos_tag = res["type"]
            word_pos_tag.append((word, pos_tag))
        wpts.append(word_pos_tag)
        return wpts


class W2vBilstmPredict:
    def __init__(self, path_dir):
        self.path_dir = path_dir
        self.tokenizer_init()
        self.l2i_i2l_init()
        self.params_init()
        self.model_init()

    def model_init(self):
        """模型初始化"""
        path_graph = os.path.join(self.path_dir, "graph.json")
        path_model = os.path.join(self.path_dir, "model.h5")
        # 加载模型结构
        self.model = model_from_json(open(path_graph, "r", encoding="utf-8").read(),
                                     custom_objects=macropodus.custom_objects)
        # 加载模型权重
        self.model.load_weights(path_model)

    def tokenizer_init(self):
        """字典"""
        # reader tokenizer
        self.token2idx = {}
        path_dict = os.path.join(self.path_dir, "vocab.txt")
        self.token2idx = load_json(path_dict)
        # with codecs.open(path_dict, 'r', 'utf8') as reader:
        #     for line in reader:
        #         token = line.strip()
        #         self.token2idx[token] = len(self.token2idx)

    def params_init(self):
        """超参数初始化"""
        # params
        path_params = os.path.join(self.path_dir, "params.json")
        self.params = load_json(path_params)
        self.len_max = self.params["len_max"]
        self.level_type = self.params["level_type"]

    def l2i_i2l_init(self):
        """类别与数字项目转化"""
        # l2i_i2l
        path_l2i_i2l = os.path.join(self.path_dir, "l2i_i2l.json")
        self.l2i_i2l = load_json(path_l2i_i2l)

    def word_encode(self, text, second_text=None):
        text = extract_chinese(str(text).upper())
        if self.level_type == 'char':
            text = list(text)
        elif self.level_type == 'word':
            text = macropodus_cut(text)
        else:
            raise RuntimeError("your input level_type is wrong, it must be 'word' or 'char'")
        text = [text_one for text_one in text]
        len_leave = self.len_max - len(text)
        if len_leave >= 0:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else 1 for
                          text_char in text] + [0 for i in range(len_leave)]
        else:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else 1 for
                          text_char in text[0:self.len_max]]
        input_mask = min(len(text), self.len_max)
        return [text_index, input_mask]

    def sentence2idx(self, text, second_text=None):
        """数据预处理"""
        text = extract_chinese(str(text).upper())
        text_embed = self.word_encode(text)
        x_ = np.array([text_embed])
        x_1 = np.array([x[0] for x in x_])
        return x_1

    def predict(self, ques):
        """预测"""
        mode_input = self.sentence2idx(ques)
        res = self.model.predict(mode_input)
        res_list = res.tolist()[0]
        res_idxs = [np.argmax(rl) for rl in res_list]
        res_label = [self.l2i_i2l["i2l"][str(ri)] if str(ri) in self.l2i_i2l["i2l"] else "O" for ri in res_idxs]
        return res_label[0:len(ques)]


if __name__ == '__main__':
    path = "macropodus/data/ner_people_1998_mix_albert_1"
    # path = "macropodus/data/tag_seg_pku_1998_w2v_16"
    # path = "macropodus/data/model/tag_people_1998_radom_bilstm_crf"

    abp = AlbertBilstmPredict(path)
    # abp = W2vBilstmPredict(path)

    while True:
        print("请输入:")
        ques = input()
        res = abp.predict([ques])
        print(res)
        res = abp.predict_single(ques)
        print(res)
