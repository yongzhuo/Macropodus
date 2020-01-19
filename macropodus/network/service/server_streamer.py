# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/1/16 22:18
# @author  : Mo
# @function: service streamer of multiprocessing


# 多进程, win10必须加, 否则报错
import platform
sys = platform.system()
if sys == "Windows":
    import multiprocessing as mp
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

from macropodus.network.service.server_base import Streamer, ThreadedStreamer
from macropodus.preprocess.tools_ml import extract_chinese
from tensorflow.python.keras.models import model_from_json
from macropodus.preprocess.tools_common import load_json
from keras_bert import Tokenizer
import numpy as np
import macropodus
import codecs
import os


# 常规
class AlbertBilstmPredict:
    def __init__(self, path_dir):
        self.path_dir = path_dir
        self.tokenizer_init()
        self.l2i_i2l_init()
        self.params_init()
        self.model_init()

    def model_init(self):
        """模型初始化"""
        # import tensorflow as tf
        # self.model = None
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)  # gpu_memory_fraction
        # config = tf.ConfigProto(gpu_options=gpu_options)
        # self.graph = tf.Graph()
        # self.sess = tf.Session(graph=self.graph, config=config)
        # with self.sess.as_default():
        #     with self.graph.as_default():
        # self.model = None
        # graph = tf.get_default_graph()
        # sess = tf.Session(graph=graph)
        # with sess.as_default():
        #     with graph.as_default():
        #         tf.global_variables_initializer().run()
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
        token_dict = {}
        path_dict = os.path.join(self.path_dir, "vocab.txt")
        with codecs.open(path_dict, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        # vocab_size = len(token_dict)
        self.tokenizer = Tokenizer(token_dict)

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
        text = extract_chinese(str(text).upper())
        input_id, input_type_id = self.tokenizer.encode(first=text, second=second_text, max_len=self.len_max)
        input_mask = len([1 for ids in input_id if ids != 0])
        return [input_id, input_type_id, input_mask]

    def predict(self, quess):
        """预测多个问句"""
        quess_encode = [self.sentence2idx(ques) for ques in quess]
        x_ = np.array(quess_encode)
        x_1 = np.array([x[0] for x in x_])
        x_2 = np.array([x[1] for x in x_])
        x_3 = np.array([x[2] for x in x_])
        ress = self.model.predict([x_1, x_2, x_3])
        ress_idxs = [[np.argmax(rl) for rl in res_list] for res_list in ress.tolist()]
        ress_label = [[self.l2i_i2l["i2l"][str(ri)] if str(ri) in self.l2i_i2l["i2l"] else "O" for ri in res_idxs]
                      for res_idxs in ress_idxs]
        ress_select = [ress_label[i][1:len(quess[i]) + 1] for i in range(len(quess))]
        return ress_select


# 一个进程多个线程&多进程等
class ServiceNer:
    def __init__(self, path_abs, cuda_devices="0", stream_type="processing",
                 max_latency=0.1, worker_num=1, batch_size=32):
        self.algorithm = 'albert-ner-bilstm-crf'
        self.cuda_devices = cuda_devices
        self.stream_type = stream_type
        self.max_latency = max_latency
        self.worker_num = worker_num
        self.batch_size = batch_size
        self.path_abs = path_abs
        self.streamer_init()

    def streamer_init(self):
        """
            ner初始化
        :param model: class, like "ner_model"
        :param cuda_devices: str, like "processing", "thread"
        :param stream_type: str, like "0,1"
        :param batch_size: int, like 32
        :param max_latency: float, 0-1, like 0.01
        :param worker_num: int, like 2
        :return: 
        """
        model = AlbertBilstmPredict(self.path_abs)
        if self.stream_type == "thread":
            self.streamer = ThreadedStreamer(model, self.batch_size, self.max_latency)
        else:
            self.streamer = Streamer(predict_function_or_model=model,
                                     cuda_devices=self.cuda_devices,
                                     max_latency=self.max_latency,
                                     worker_num=self.worker_num,
                                     batch_size=self.batch_size)
            self.streamer._wait_for_worker_ready()

    # def predict(self, text):
    #     """
    #         预测返回
    #     :param text: str, like "桂林"
    #     :return: list, like ["B-LOC", "I-LOC"]
    #     """
    #     return self.streamer.predict(text)


# 模型加载
# path = "D:/workspace/pythonMyCode/Macropodus/macropodus/data/tag_seg_pku_1998_w2v_16"
path = "D:/workspace/pythonMyCode/Macropodus/macropodus/data/ner_people_1998_mix_albert_1"
model_server = ServiceNer(path, stream_type="thread", cuda_devices="-1", max_latency=0.1, worker_num=1, batch_size=32).streamer


if __name__ == '__main__':
    ques = "北京欢迎您, 南宁2020东盟博览会"
    res = model_server.predict([ques])
    print(res)
    while True:
        print("请输入:")
        ques = input()
        res = model_server.predict([ques])
        print(res)
