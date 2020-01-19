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
        self.filters = hyper_parameters['model'].get('filters', [2, 3, 4])
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
        embedding = self.word_embedding.output
        # 提取n-gram特征和最大池化， 一般不用平均池化
        conv_pools = [embedding]
        for filter in self.filters:
            conv = tf.keras.layers.Conv1D(filters=self.filters_num,
                                          kernel_size=filter,
                                          padding='same',
                                          kernel_initializer='normal',
                                          activation='relu', )(embedding)
            pooled = tf.keras.layers.MaxPool1D(pool_size=2,
                                               strides=1,
                                               padding='same', )(conv)
            conv_pools.append(pooled)
        # 拼接
        x = tf.keras.layers.Concatenate(axis=-1)(conv_pools)
        # Bi-LSTM
        for nrl in range(self.num_rnn_layers):
            x = tf.keras.layers.Bidirectional(self.rnn_layer(units=self.rnn_units,
                                             return_sequences=True,
                                             activation=self.activate_rnn,
                                             kernel_regularizer=tf.keras.regularizers.l2(self.l2),
                                             recurrent_regularizer=tf.keras.regularizers.l2(self.l2)
                                             ))(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(self.label, activation=self.activate_classify, name='layer_dense_3')(x)
        self.output = x
        self.model = tf.keras.Model(self.word_embedding.input, self.output)
        self.model.summary(132)
