# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/1/9 21:44
# @author  : Mo
# @function: CRF


from macropodus.network.base.graph import graph
from macropodus.network.layers.crf import CRF
import tensorflow as tf


class CRFGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.crf_mode = hyper_parameters["model"].get("crf_mode", "reg") # "reg", pad
        self.supports_masking = hyper_parameters["model"].get("supports_masking", True) # True or False
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        x = self.word_embedding.output
        # TimeDistributed
        x_64 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation="softmax"),
                                                 name='layer_time_distributed')(x)
        # dense to a smaller units
        tensor = tf.keras.layers.Dense(units=self.label, activation=self.activate_rnn, name="layer_dense_64")(x_64)
        # crf, "pad" or "reg"
        if self.crf_mode == "pad":
            # length of real sentence
            x_mask = tf.keras.layers.Input(shape=(1), dtype=tf.int32)
            self.crf = CRF(self.label, mode="pad", supports_masking=True, name="layer_crf")
            self.output = self.crf([tensor, x_mask])
            if self.embedding_type in ["bert", "albert"]:
                self.inputs = [self.word_embedding.input[0], self.word_embedding.input[1], x_mask]
            else:
                self.inputs = [self.word_embedding.input, x_mask]
        else:
            self.crf = CRF(self.label, mode="reg", name="layer_crf")
            self.output = self.crf(tensor)
            self.inputs = self.word_embedding.input
        self.model = tf.keras.Model(self.inputs, self.output)
        self.model.summary(132)

    def create_compile(self):
        """
          构建优化器、损失函数和评价函数
        :return: 
        """
        self.loss = self.crf.loss
        self.metrics = self.crf.viterbi_accuracy
        super().create_compile()
