# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/20 22:22
# @author  : Mo
# @function: init of keras of tensorflow


#####################(tensorflow, keras)############################
import sys
import os
path_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(path_root) # 环境引入根目录
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_KERAS'] = '1'

try:
    # tensorflow.python.keras
    import tensorflow.python.keras as keras
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    import keras

# custom_objects
import keras_bert
custom_objects = keras_bert.get_custom_objects()
from keras_adaptive_softmax import AdaptiveEmbedding, AdaptiveSoftmax
from macropodus.network.layers.non_mask_layer import NonMaskingLayer
from macropodus.network.layers.crf import CRF
custom_objects['AdaptiveEmbedding'] = AdaptiveEmbedding
custom_objects['AdaptiveSoftmax'] = AdaptiveSoftmax
custom_objects['NonMaskingLayer'] = NonMaskingLayer
custom_objects['CRF'] = CRF

# layers
preprocessing = keras.preprocessing
applications = keras.applications
regularizers = keras.regularizers
initializers = keras.initializers
activations = keras.activations
constraints = keras.constraints
optimizers = keras.optimizers
callbacks = keras.callbacks
datasets = keras.datasets
wrappers = keras.wrappers
metrics = keras.metrics
backend = keras.backend
engine = keras.engine
layers = keras.layers
models = keras.models
losses = keras.losses
utils = keras.utils
