# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/11/10 21:35
# @author   :Mo
# @function :NonMaskingLayer of bert
# @codefrom :https://github.com/jacoxu


from __future__ import print_function, division
from tensorflow.python.keras.layers import Layer


class NonMaskingLayer(Layer):
    """
    fix convolutional 1D can't receive masked input,
    detail: https://github.com/keras-team/keras/issues/4978
    """

    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def compute_mask(self, inputs, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x
