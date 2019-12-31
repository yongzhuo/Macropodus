# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/20 23:08
# @author  : Mo
# @function: predict(albert+bilstm)


from macropodus.conf.path_config import path_embedding_albert
from macropodus.preprocess.tools_ml import extract_chinese
from tensorflow.python.keras.models import model_from_json
from macropodus.preprocess.tools_common import load_json
from macropodus.conf.path_config import path_model_dir
from keras_bert import Tokenizer
import numpy as np
import macropodus
import codecs
import pickle
import os


path_dir = path_model_dir # + "/ner_albert_bilstm_people_199801"
# 加载模型结构
model = model_from_json(open(path_dir+"/graph.json", "r", encoding="utf-8").read(),
                        custom_objects=macropodus.custom_objects)
# 加载模型权重
model.load_weights(path_dir+"/model.h5")

# reader tokenizer
token_dict = {}
path_dict = os.path.join(path_embedding_albert, "vocab.txt")
with codecs.open(path_dict, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)
vocab_size = len(token_dict)
tokenizer = Tokenizer(token_dict)
# params
path_params = path_dir + "/params.json"
params = load_json(path_params)
len_max = params["len_max"]
# l2i_i2l
path_l2i_i2l = path_dir + "/l2i_i2l.json"
l2i_i2l = load_json(path_l2i_i2l)

def sentence2idx(text, second_text=None):
    text = extract_chinese(str(text).upper())
    input_id, input_type_id = tokenizer.encode(first=text, second=second_text, max_len=len_max)
    input_mask = len([1 for ids in input_id if ids != 0])
    # return input_id, input_type_id, input_mask
    # x_ = np.array((input_id, input_type_id, input_mask))
    x = [[input_id, input_type_id, input_mask]]
    x_ = np.array(x)
    x_1 = np.array([x[0] for x in x_])
    x_2 = np.array([x[1] for x in x_])
    x_3 = np.array([x[2] for x in x_])

    return [x_1, x_2, x_3]

while True:
    print("请输入:")
    ques = input()
    mode_input = sentence2idx(ques)
    res = model.predict(mode_input)
    res_list = res.tolist()[0]
    res_idxs = [np.argmax(rl) for rl in res_list]
    res_label = [l2i_i2l["i2l"][str(ri)] if str(ri) in l2i_i2l["i2l"] else "O" for ri in  res_idxs]
    print(res_label[:len(ques)])

# gg = 0

# # 保存模型的结构
# json_string = model.to_json()  # 方式1
# open("model_architecture_1.json", "w").write(json_string)
# yaml_string = model.to_yaml()  # 方式2
# open("model_arthitecture_2.yaml", "w").write(yaml_string)
# # 加载模型结构
# model = model_from_json(open("model_architecture_1.json", "r").read())
# # 加载模型权重
# model.load_weights("weights-improvement-40-0.96208.hdf5")
# # 编译模型
# model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])