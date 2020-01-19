# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/20 22:22
# @author  : Mo
# @function: init of keras of tensorflow


try:
    #####################(tensorflow, keras)############################
    import sys
    import os

    path_root = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(path_root)  # 环境引入根目录
    # 默认cpu环境, tensorflow
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_KERAS'] = '1'

    # tensorflow.python.keras
    from macropodus.network.service.server_prdeict import AlbertBilstmPredict
    from keras_adaptive_softmax import AdaptiveEmbedding, AdaptiveSoftmax
    from macropodus.network.layers.non_mask_layer import NonMaskingLayer
    from macropodus.conf.path_config import path_model_dir
    from macropodus.network.layers.crf import CRF
    import tensorflow.python.keras as keras
    import tensorflow as tf
    import keras_bert

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # custom_objects
    custom_objects = keras_bert.get_custom_objects()
    custom_objects['AdaptiveEmbedding'] = AdaptiveEmbedding
    custom_objects['AdaptiveSoftmax'] = AdaptiveSoftmax
    custom_objects['NonMaskingLayer'] = NonMaskingLayer
    custom_objects['CRF'] = CRF

    # init model of dl(deep learning)
    # 加载训练好的模型, 命名实体提取
    path_ner_albert_bilstm_crf = os.path.join(path_model_dir, 'ner_albert_people_1998')
    ner_albert_bilstm_crf = AlbertBilstmPredict(path_ner_albert_bilstm_crf, custom_objects)
    ner = ner_albert_bilstm_crf.predict_single
    ners = ner_albert_bilstm_crf.predict

    # # layers
    # preprocessing = keras.preprocessing
    # applications = keras.applications
    # regularizers = keras.regularizers
    # initializers = keras.initializers
    # activations = keras.activations
    # constraints = keras.constraints
    # optimizers = keras.optimizers
    # callbacks = keras.callbacks
    # datasets = keras.datasets
    # wrappers = keras.wrappers
    # metrics = keras.metrics
    # backend = keras.backend
    # engine = keras.engine
    # layers = keras.layers
    # models = keras.models
    # losses = keras.losses
    # utils = keras.utils
except Exception as e:
    from macropodus.conf.path_log import get_logger_root
    logger = get_logger_root()
    logger.info(str(e))
