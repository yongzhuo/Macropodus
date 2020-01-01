# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/12/3 20:51
# @author   :Mo
# @function :graph of base


from macropodus.network.preprocess.preprocess_generator import PreprocessGenerator
from macropodus.network.layers.keras_lookahead import Lookahead
from macropodus.preprocess.tools_common import save_json
from macropodus.network.layers.keras_radam import RAdam
from macropodus.conf.path_config import path_model_dir
from macropodus.conf.path_log import get_logger_root
import tensorflow as tf
import numpy as np
import os


logger = get_logger_root()


class graph:
    def __init__(self, hyper_parameters):
        """
            模型初始化
        :param hyper_parameters:json, json["model"] and json["embedding"]  
        """
        self.len_max = hyper_parameters.get("len_max", 50)  # 文本最大长度
        self.embed_size = hyper_parameters.get("embed_size", 300)  # 嵌入层尺寸
        self.trainable = hyper_parameters.get("trainable", False)  # 是否微调, 例如静态词向量、动态词向量、微调bert层等, random也可以
        self.embedding_type = hyper_parameters.get("embedding_type", "word2vec")  # 词嵌入方式，可以选择"xlnet"、"bert"、"gpt-2"、"word2vec"或者"None"
        self.gpu_memory_fraction = hyper_parameters.get("gpu_memory_fraction", None)  # gpu使用率, 默认不配置
        self.hyper_parameters = hyper_parameters
        hyper_parameters_model = hyper_parameters["model"]
        self.label = hyper_parameters_model.get("label", 2)  # 类型
        self.batch_size = hyper_parameters_model.get("batch_size", 32)  # 批向量
        self.filters = hyper_parameters_model.get("filters", [3, 4, 5])  # 卷积核大小
        self.filters_num = hyper_parameters_model.get("filters_num", 300)  # 核数
        self.channel_size = hyper_parameters_model.get("channel_size", 1)  # 通道数
        self.dropout = hyper_parameters_model.get("dropout", 0.5)  # dropout层系数，舍弃
        self.decay_step = hyper_parameters_model.get("decay_step", 100)  # 衰减步数
        self.decay_rate = hyper_parameters_model.get("decay_rate", 0.9)  # 衰减系数
        self.epochs = hyper_parameters_model.get("epochs", 20)  # 训练轮次
        self.vocab_size = hyper_parameters_model.get("vocab_size", 20000)  # 字典词典大小
        self.lr = hyper_parameters_model.get("lr", 1e-3)  # 学习率
        self.l2 = hyper_parameters_model.get("l2", 1e-6)  # l2正则化系数
        self.activate_rnn = hyper_parameters_model.get("activate_rnn", "tanh")  # RNN激活函数, tanh, relu, signmod
        self.activate_classify = hyper_parameters_model.get("activate_classify", "softmax")  # 分类激活函数,softmax或者signmod
        self.loss = hyper_parameters_model.get("loss", "categorical_crossentropy")  # 损失函数, mse, categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy等
        self.metrics = hyper_parameters_model.get("metrics", "accuracy")  # acc, binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, sparse_top_k_categorical_accuracy
        self.is_training = hyper_parameters_model.get("is_training", False)  # 是否训练, 保存时候为Flase,方便预测
        self.patience = hyper_parameters_model.get("patience", 3)  # 早停, 2-3就可以了
        self.optimizer_name = hyper_parameters_model.get("optimizer_name", "RAdam,Lookahead")  # 早停, 2-3就可以了
        self.path_model_dir = hyper_parameters_model.get("path_model_dir", path_model_dir)  # 模型目录
        self.path_fineture = os.path.join(self.path_model_dir, "embedding_trainable.h5")  # embedding层保存地址, 例如静态词向量、动态词向量、微调bert层等
        self.path_model = os.path.join(self.path_model_dir, "model.h5")  # 模型weight绝对地址
        self.path_hyper_parameters = os.path.join(self.path_model_dir, "params.json")  # 超参数保存绝对地址
        self.path_model_l2i_i2l = os.path.join(self.path_model_dir, "l2i_i2l.json")  # 模型字典保存绝对地址
        self.path_model_graph =  os.path.join(self.path_model_dir, "graph.json")  # 模型图结构绝对地址
        if self.gpu_memory_fraction:
            # keras, tensorflow控制GPU使用率等
            import tensorflow.python.keras.backend as K
            import tensorflow as tf
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
            # config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            K.set_session(sess)
        self.create_model(hyper_parameters) # 模型初始化
        if self.is_training:  # 是否是训练阶段, 与预测区分开
            self.create_compile()
            self.save_graph()  # 保存图结构

    def create_model(self, hyper_parameters):
        """
            构建神经网络
        :param hyper_parameters: json，超参数
        :return:  
        """
        # embeddings选择
        if self.embedding_type == "albert":
            from macropodus.network.base.embedding import AlbertEmbedding as Embeddings
        elif self.embedding_type == "random":
            from macropodus.network.base.embedding import RandomEmbedding as Embeddings
        elif self.embedding_type == "word2vec":
            from macropodus.network.base.embedding import WordEmbedding as Embeddings
        elif self.embedding_type == "bert":
            from macropodus.network.base.embedding import BertEmbedding as Embeddings
        else:
            raise RuntimeError("your input embedding_type is wrong, it must be 'random'、 'bert'、 'albert' or 'word2vec'")
        # 构建网络层
        self.word_embedding = Embeddings(hyper_parameters=hyper_parameters)
        if os.path.exists(self.path_fineture) and self.trainable:
            self.word_embedding.model.load_weights(self.path_fineture)
            print("load path_fineture ok!")
        self.model = None

    def callback(self):
        """
          评价函数、早停
        :return: callback
        """
        # import datetime
        # self.path_model_dir = os.path.join(self.path_model_dir, "plugins/profile", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        cb_em = [tf.keras.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", filepath=self.path_model, verbose=1, save_best_only=True, save_weights_only=False),
                 tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.path_model_dir, "logs"), batch_size=self.batch_size, update_freq='batch'),
                 tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", min_delta=1e-8, patience=self.patience),
                 ]
        return cb_em

    def create_compile(self):
        """
          构建优化器、损失函数和评价函数
        :return: 
        """
        if self.optimizer_name.upper() == "ADAM":
            self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                               loss=self.loss,
                               metrics=[self.metrics])  # Any optimize
        elif self.optimizer_name.upper() == "RADAM":
            self.model.compile(optimizer=RAdam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                               loss=self.loss,
                               metrics=[self.metrics])  # Any optimize
        else:
            self.model.compile(optimizer=RAdam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0),
                               loss=self.loss,
                               metrics=[self.metrics])  # Any optimize
            lookahead = Lookahead(k=5, alpha=0.5)  # Initialize Lookahead
            lookahead.inject(self.model)  # add into model

    def fit(self, x_train, y_train, x_dev, y_dev):
        """
            训练
        :param x_train: 
        :param y_train: 
        :param x_dev: 
        :param y_dev: 
        :return: 
        """
        # 保存超参数
        self.hyper_parameters["model"]["is_training"] = False  # 预测时候这些设为False
        self.hyper_parameters["model"]["trainable"] = False
        self.hyper_parameters["model"]["dropout"] = 1.0

        save_json(json_lines=self.hyper_parameters, json_path=self.path_hyper_parameters)
        # 训练模型
        self.model.fit(x_train, y_train, batch_size=self.batch_size,
                       epochs=self.epochs, validation_data=(x_dev, y_dev),
                       shuffle=True,
                       callbacks=self.callback())
        # 保存embedding, 动态的
        if self.trainable:
            self.word_embedding.model.save(self.path_fineture)

    def fit_generator(self, embed, rate=1):
        """

        :param data_fit_generator: yield, 训练数据
        :param data_dev_generator: yield, 验证数据
        :param steps_per_epoch: int, 训练一轮步数
        :param validation_steps: int, 验证一轮步数
        :return: 
        """
        # 保存超参数
        self.hyper_parameters["model"]["is_training"] = False  # 预测时候这些设为False
        self.hyper_parameters["model"]["trainable"] = False
        self.hyper_parameters["model"]["dropout"] = 1.0

        save_json(json_lines=self.hyper_parameters, json_path=self.path_hyper_parameters)

        pg = PreprocessGenerator(self.path_model_l2i_i2l)
        _, len_train = pg.preprocess_label2set(self.hyper_parameters["data"]["train_data"])
        data_fit_generator = pg.preprocess_label_question_to_idx_fit_generator(embedding_type=self.hyper_parameters["embedding_type"],
                                                                               crf_mode=self.hyper_parameters["model"]["crf_mode"],
                                                                               path=self.hyper_parameters["data"]["train_data"],
                                                                               batch_size=self.batch_size,
                                                                               embed=embed,
                                                                               rate=rate)
        _, len_val = pg.preprocess_label2set(self.hyper_parameters["data"]["val_data"])
        data_dev_generator = pg.preprocess_label_question_to_idx_fit_generator(embedding_type=self.hyper_parameters["embedding_type"],
                                                                               crf_mode=self.hyper_parameters["model"]["crf_mode"],
                                                                               path=self.hyper_parameters["data"]["val_data"],
                                                                               batch_size=self.batch_size,
                                                                               embed=embed,
                                                                               rate=rate)
        steps_per_epoch = len_train // self.batch_size
        validation_steps = len_val // self.batch_size
        # 训练模型
        self.model.fit_generator(generator=data_fit_generator,
                                 validation_data=data_dev_generator,
                                 callbacks=self.callback(),
                                 epochs=self.epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps)
        # 保存embedding, 动态的
        if self.trainable:
            self.word_embedding.model.save(self.path_fineture)

    def save_graph(self):
        """
            模型图保存
        :return: None
        """
        # # 序列化模型embidding
        # import pickle
        # file_fineture = open(self.path_fineture, "wb")
        # pickle.dumps(self.word_embedding.sentence2idx, file_fineture)
        # 序列化模型graph
        json_string = self.model.to_json()
        open(self.path_model_graph, "w", encoding="utf-8").write(json_string)

    def load_model(self):
        """
          模型下载
        :return: None
        """
        logger.info("load_model start!")
        self.model.load_weights(self.path_model)
        logger.info("load_model end!")

    def predict(self, sen):
        """
          预测
        :param sen: 
        :return: 
        """
        if self.embedding_type in ["bert", "albert"]:
            if type(sen) == np.ndarray:
                sen = sen.tolist()
            elif type(sen) == list:
                sen = sen
            else:
                raise RuntimeError("your input sen is wrong, it must be type of list or np.array")
            return self.model.predict(sen)
        else:
            if type(sen) == np.ndarray:
                sen = sen
            elif type(sen) == list:
                sen = np.array([sen])
            else:
                raise RuntimeError("your input sen is wrong, it must be type of list or np.array")
            return self.model.predict(sen)
