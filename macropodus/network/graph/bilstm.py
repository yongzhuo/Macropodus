# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/19 22:30
# @author  : Mo
# @function: Bi-LSTM


from macropodus.network.base.graph import graph
import tensorflow as tf


class BiLSTMGraph(graph):
    def __init__(self, hyper_parameters):
        """
            初始化
        :param hyper_parameters: json，超参
        """
        self.num_rnn_layers = hyper_parameters['model'].get('num_rnn_layers', 1)
        self.rnn_type = hyper_parameters['model'].get('rnn_type', 'LSTM')
        self.rnn_units = hyper_parameters['model'].get('rnn_units', 256)
        super().__init__(hyper_parameters)

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters:json,  hyper parameters of network
        :return: tensor, moedl
        """
        super().create_model(hyper_parameters)
        self.rnn_layer = {'LSTM':tf.keras.layers.LSTM, 'GRU':tf.keras.layers.GRU}[self.rnn_type]
        x = self.word_embedding.output
        # Bi-LSTM
        for nrl in range(self.num_rnn_layers):
            x = tf.keras.layers.Bidirectional(self.rnn_layer(units=self.rnn_units,
                                         return_sequences=True,
                                         activation=self.activate_rnn,
                                         kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                                         recurrent_regularizer=tf.keras.regularizers.l2(self.l2)
                                         ))(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
        x_time = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.label, name='layer_crf_dense'))(x)
        x_act = tf.keras.layers.Activation(activation=self.activate_classify)(x_time)
        self.output = x_act
        self.model = tf.keras.Model(self.word_embedding.input, self.output)
        self.model.summary(132)
