# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/6 20:45
# @author  : Mo
# @function: Bi-LSTM-CRF


from macropodus.network.layers.keras_lookahead import Lookahead
from macropodus.network.layers.keras_radam import RAdam
from macropodus.network.base.graph import graph
from macropodus.network.layers.crf import CRF
import tensorflow as tf


class BilstmCRFGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.num_rnn_layers = hyper_parameters['model'].get('num_rnn_layers', 1) # 1, 2, 3
        self.rnn_type = hyper_parameters['model'].get('rnn_type', 'LSTM') # 'LSTM', 'GRU'
        self.rnn_units = hyper_parameters['model'].get('rnn_units', 512) # 128, 256, 512, 768, 1024
        self.crf_mode = hyper_parameters['model'].get('crf_mode', 'reg') # "reg", pad
        self.supports_masking = hyper_parameters['model'].get('supports_masking', True) # True or False
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        # LSTM or GRU
        self.rnn_layer = {'LSTM':tf.keras.layers.LSTM, 'GRU':tf.keras.layers.GRU}[self.rnn_type]
        x = self.word_embedding.output
        # Bi-LSTM
        for nrl in range(self.num_rnn_layers):
            x = tf.keras.layers.Bidirectional(self.rnn_layer(units=self.rnn_units,
                                         return_sequences=True,
                                         activation=self.activate_rnn,
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2 * 0.1),
                                         recurrent_regularizer=tf.keras.regularizers.l2(self.l2)
                                         ))(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
        # crf, 'pad' or 'reg'
        if self.crf_mode == "pad":
            # length of real sentence
            x_mask = tf.keras.layers.Input(shape=(1), dtype=tf.int32)
            self.crf = CRF(self.label, mode='pad', supports_masking=True, name='crf')
            tensor = tf.keras.layers.Dense(self.label, name='crf_dense')(x)
            self.output = self.crf([tensor, x_mask])
            if self.embedding_type in ["bert", "albert"]:
                self.inputs = [self.word_embedding.input[0], self.word_embedding.input[1], x_mask]
            else:
                self.inputs = [self.word_embedding.input, x_mask]
        else:
            self.crf = CRF(self.label, mode='reg', name='crf')
            tensor = tf.keras.layers.Dense(self.label, name='crf_dense')(x)
            self.output = self.crf(tensor)
            if self.embedding_type in ["bert", "albert"]:
                self.inputs = self.word_embedding.input
            else:
                self.inputs = self.word_embedding.input
        self.model = tf.keras.Model(self.inputs, self.output)
        self.model.summary(132)

    def create_compile(self):
        """
          构建优化器、损失函数和评价函数
        :return: 
        """
        if self.optimizer_name.upper() == "ADAM":
            self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                               loss=self.crf.loss,
                               metrics=[self.crf.viterbi_accuracy])  # Any optimize, [self.metrics])
        elif self.optimizer_name.upper() == "RADAM":
            self.model.compile(optimizer=RAdam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                               loss=self.crf.loss,
                               metrics=[self.crf.viterbi_accuracy]) # Any optimize
        else:
            self.model.compile(optimizer=RAdam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                               loss=self.crf.loss,
                               metrics=[self.crf.viterbi_accuracy]) # Any optimize
            lookahead = Lookahead(k=5, alpha=0.5)  # Initialize Lookahead
            lookahead.inject(self.model)  # add into model
