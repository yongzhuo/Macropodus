# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/11/3 21:29
# @author   :Mo
# @function :embeddings of model, base embedding of random, word2vec or bert


from macropodus.preprocess.tools_ml import extract_chinese, macropodus_cut, get_ngrams, gram_uni_bi_tri
from macropodus.preprocess.tools_common import save_json, load_json, txt_read, txt_write
from macropodus.conf.path_config import path_embedding_bert, path_embedding_albert
from macropodus.network.layers.non_mask_layer import NonMaskingLayer
from macropodus.conf.path_config import path_embedding_word2vec_char
from macropodus.conf.path_config import path_embedding_random_char
from macropodus.conf.path_config import path_model_dir
from macropodus.conf.path_log import get_logger_root
from gensim.models import KeyedVectors
import tensorflow as tf
import numpy as np
import codecs
import os


logger = get_logger_root()


class BaseEmbedding:
    def __init__(self, hyper_parameters):
        self.len_max = hyper_parameters.get('len_max', 50)  # 文本最大长度, 建议25-50
        self.embed_size = hyper_parameters.get('embed_size', 300)  # 嵌入层尺寸
        self.vocab_size = hyper_parameters.get('vocab_size', 30000)  # 字典大小, 这里随便填的，会根据代码里修改
        self.trainable = hyper_parameters.get('trainable', False)  # 是否微调, 例如静态词向量、动态词向量、微调bert层等, random也可以
        self.level_type = hyper_parameters.get('level_type', 'char')  # 还可以填'word'
        self.embedding_type = hyper_parameters.get('embedding_type', 'word2vec')  # 词嵌入方式，可以选择'xlnet'、'bert'、'random'、'word2vec'
        self.path_model_dir = hyper_parameters.get('model', {}).get("path_model_dir", path_model_dir)  # 模型目录, 提供给字/词典

        # 自适应, 根据level_type和embedding_type判断corpus_path
        if self.level_type == "word":
            if self.embedding_type == "random":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_random_char)
            elif self.embedding_type == "word2vec":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_word2vec_char)
            elif self.embedding_type == "bert":
                raise RuntimeError("bert level_type is 'char', not 'word'")
            elif self.embedding_type == "xlnet":
                raise RuntimeError("xlnet level_type is 'char', not 'word'")
            elif self.embedding_type == "albert":
                raise RuntimeError("albert level_type is 'char', not 'word'")
            else:
                raise RuntimeError("embedding_type must be 'random', 'word2vec' or 'bert'")
        elif self.level_type == "char":
            if self.embedding_type == "random":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_random_char)
            elif self.embedding_type == "word2vec":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_word2vec_char)
            elif self.embedding_type == "bert":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_bert)
            elif self.embedding_type == "albert":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path', path_embedding_albert)
            else:
                raise RuntimeError("embedding_type must be 'random', 'word2vec' or 'bert'")
        elif self.level_type == "ngram":
            if self.embedding_type == "random":
                self.corpus_path = hyper_parameters['embedding'].get('corpus_path')
                if not self.corpus_path:
                    raise RuntimeError("corpus_path must exists!")
            else:
                raise RuntimeError("embedding_type must be 'random', 'word2vec' or 'bert'")
        else:
            raise RuntimeError("level_type must be 'char' or 'word'")
        # 定义的符号
        self.ot_dict = {'[PAD]': 0,
                        '[UNK]': 1,
                        '[BOS]': 2,
                        '[EOS]': 3, }
        self.deal_corpus()
        self.build()
        logger.info("Embedding init ok!")

    def deal_corpus(self):  # 处理语料
        pass

    def build(self):
        self.token2idx = {}
        self.idx2token = {}

    def sentence2idx(self, text):
        text = extract_chinese(str(text).upper())
        if self.level_type == 'char':
            text = list(text)
        elif self.level_type == 'word':
            text = macropodus_cut(text)
        else:
            raise RuntimeError("your input level_type is wrong, it must be 'word' or 'char'")
        text = [text_one for text_one in text]
        len_leave = self.len_max - len(text)
        if len_leave >= 0:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text] + [self.token2idx['[PAD]'] for i in range(len_leave)]
        else:
            text_index = [self.token2idx[text_char] if text_char in self.token2idx else self.token2idx['[UNK]'] for
                          text_char in text[0:self.len_max]]
        input_mask = min(len(text), self.len_max)
        return [text_index, input_mask]

    def idx2sentence(self, idx):
        assert type(idx) == list
        text_idx = [self.idx2token[id] if id in self.idx2token else self.idx2token['[UNK]'] for id in idx]
        return "".join(text_idx)


class RandomEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        self.ngram_ns = hyper_parameters['embedding'].get('ngram_ns', [1, 2, 3]) # ngram信息, 根据预料获取
        super().__init__(hyper_parameters)
        # self.path = hyper_parameters.get('corpus_path', path_embedding_random_char)

    def deal_corpus(self):
        import json

        token2idx = self.ot_dict.copy()
        if 'term' in self.corpus_path:
            with open(file=self.corpus_path, mode='r', encoding='utf-8') as fd:
                while True:
                    term_one = fd.readline()
                    if not term_one:
                        break
                    if term_one not in token2idx:
                        token2idx[term_one] = len(token2idx)
        elif os.path.exists(self.corpus_path):
            with open(file=self.corpus_path, mode='r', encoding='utf-8') as fd:
                terms = fd.readlines()
                for line in terms:
                    ques_label = json.loads(line.strip())
                    term_one = ques_label["question"]
                    term_one = "".join(term_one)
                    if self.level_type == 'char':
                        text = list(term_one.replace(' ', '').strip())
                    elif self.level_type == 'word':
                        text = macropodus_cut(term_one)
                    elif self.level_type == 'ngram':
                        text = get_ngrams(term_one, ns=self.ngram_ns)
                    else:
                        raise RuntimeError("your input level_type is wrong, it must be 'word', 'char', 'ngram'")
                    for text_one in text:
                        if term_one not in token2idx:
                            token2idx[text_one] = len(token2idx)
        else:
            raise RuntimeError("your input corpus_path is wrong, it must be 'dict' or 'corpus'")
        self.token2idx = token2idx
        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.idx2token[value] = key

    def build(self, **kwargs):
        self.vocab_size = len(self.token2idx)
        logger.info("vocab_size is {}".format(str(self.vocab_size)))
        self.input = tf.keras.layers.Input(shape=(self.len_max,), dtype='int32', name="input")
        self.output = tf.keras.layers.Embedding(self.vocab_size+1,
                                                self.embed_size,
                                                input_length=self.len_max,
                                                trainable=self.trainable,
                                                name="embedding_{}".format(str(self.embed_size)))(self.input)
        self.model = tf.keras.Model(self.input, self.output)
        save_json(json_lines=self.token2idx, json_path=os.path.join(self.path_model_dir, 'vocab.txt'))


class WordEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        super().__init__(hyper_parameters)
        # self.path = hyper_parameters.get('corpus_path', path_embedding_vector_word2vec)

    def build(self, **kwargs):
        self.embedding_type = 'word2vec'
        # logger.info("load word2vec start!")
        self.key_vector = KeyedVectors.load_word2vec_format(self.corpus_path, **kwargs)
        # logger.info("load word2vec end!")
        self.embed_size = self.key_vector.vector_size

        self.token2idx = self.ot_dict.copy()
        embedding_matrix = []
        # 首先加self.token2idx中的四个[PAD]、[UNK]、[BOS]、[EOS]
        embedding_matrix.append(np.zeros(self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))
        embedding_matrix.append(np.random.uniform(-0.5, 0.5, self.embed_size))

        for word in self.key_vector.index2entity:
            self.token2idx[word] = len(self.token2idx)
            embedding_matrix.append(self.key_vector[word])

        # self.token2idx = self.token2idx
        self.idx2token = {}
        for key, value in self.token2idx.items():
            self.idx2token[value] = key

        self.vocab_size = len(self.token2idx)
        logger.info("vocab_size is {}".format(str(self.vocab_size)))
        embedding_matrix = np.array(embedding_matrix)
        # self.input = Input(shape=(self.len_max,), dtype='int32')
        self.input = tf.keras.layers.Input(shape=(self.len_max,), dtype='int32', name="input")

        self.output = tf.keras.layers.Embedding(self.vocab_size,
                                                self.embed_size,
                                                input_length=self.len_max,
                                                weights=[embedding_matrix],
                                                trainable=self.trainable,
                                                name="embedding_{}".format(str(self.embed_size)))(self.input)
        self.model = tf.keras.Model(self.input, self.output)
        # 保存字/词典
        save_json(json_lines=self.token2idx, json_path=os.path.join(self.path_model_dir, 'vocab.txt'))


class BertEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        self.layer_indexes = hyper_parameters['embedding'].get('layer_indexes', [12])
        super().__init__(hyper_parameters)

    def build(self):
        import keras_bert

        self.embedding_type = 'bert'
        config_path = os.path.join(self.corpus_path, 'bert_config.json')
        check_point_path = os.path.join(self.corpus_path, 'bert_model.ckpt')
        dict_path = os.path.join(self.corpus_path, 'vocab.txt')
        # logger.info('load bert model start!')
        model = keras_bert.load_trained_model_from_checkpoint(config_path,
                                                              check_point_path,
                                                              seq_len=self.len_max,
                                                              trainable=self.trainable)
        # logger.info('load bert model success!')
        # bert model all layers
        layer_dict = [6]
        layer_0 = 7
        for i in range(12):
            layer_0 = layer_0 + 8
            layer_dict.append(layer_0)
        logger.info(layer_dict)
        # 输出它本身
        if len(self.layer_indexes) == 0:
            encoder_layer = model.output
        # 分类如果只有一层，就只取最后那一层的weight；取得不正确，就默认取最后一层
        elif len(self.layer_indexes) == 1:
            if self.layer_indexes[0] in [i + 1 for i in range(13)]:
                encoder_layer = model.get_layer(index=layer_dict[self.layer_indexes[0] - 1]).output
            else:
                encoder_layer = model.get_layer(index=layer_dict[-1]).output
        # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
        else:
            all_layers = [model.get_layer(index=layer_dict[lay - 1]).output if lay in [i + 1 for i in range(13)]
                          else model.get_layer(index=layer_dict[-1]).output  # 如果给出不正确，就默认输出最后一层
                          for lay in self.layer_indexes]
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = tf.keras.layers.Add(name="layer_add_bert")(all_layers_select)
        self.output = NonMaskingLayer(name="layer_non_masking_layer")(encoder_layer)
        self.input = model.inputs
        self.model = tf.keras.Model(self.input, self.output)

        self.embedding_size = self.model.output_shape[-1]

        # reader tokenizer
        self.token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        self.vocab_size = len(self.token_dict)
        self.tokenizer = keras_bert.Tokenizer(self.token_dict)

    def sentence2idx(self, text, second_text=None):
        text = extract_chinese(str(text).upper())
        input_id, input_type_id = self.tokenizer.encode(first=text, second=second_text, max_len=self.len_max)
        input_mask = len([1 for ids in input_id if ids == 1])
        return [input_id, input_type_id, input_mask]
        # input_mask = [0 if ids == 0 else 1 for ids in input_id]
        # return input_id, input_type_id, input_mask
        # return input_id, input_type_id


class AlbertEmbedding(BaseEmbedding):
    def __init__(self, hyper_parameters):
        self.layer_indexes = hyper_parameters['embedding'].get('layer_indexes', [12])
        super().__init__(hyper_parameters)

    def build(self):
        from macropodus.network.layers.albert import load_brightmart_albert_zh_checkpoint
        import keras_bert

        self.embedding_type = 'albert'
        dict_path = os.path.join(self.corpus_path, 'vocab.txt')
        # logger.info('load albert model start!')
        layer_real  = [i for i in range(25)] + [-i for i in range(25)]
        # 简要判别一下
        self.layer_indexes = [i if i in layer_real else -2 for i in self.layer_indexes]
        self.model = load_brightmart_albert_zh_checkpoint(self.corpus_path,
                                                         training=self.trainable,
                                                         seq_len=self.len_max,
                                                         output_layers = None) # self.layer_indexes)
        # model_l = self.model.layers
        # logger.info('load albert model success!')
        # albert model all layers
        layer_dict = [4, 8, 11, 13]
        layer_0 = 13
        for i in range(20):
            layer_0 = layer_0 + 1
            layer_dict.append(layer_0)
        layer_dict.append(34)
        logger.info(layer_dict)
        # 输出它本身
        if len(self.layer_indexes) == 0:
            encoder_layer = self.model.output
        # 分类如果只有一层，就只取最后那一层的weight；取得不正确，就默认取最后一层
        elif len(self.layer_indexes) == 1:
            if self.layer_indexes[0] in layer_real:
                encoder_layer = self.model.get_layer(index=layer_dict[self.layer_indexes[0]]).output
            else:
                encoder_layer = self.model.get_layer(index=layer_dict[-2]).output
        # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
        else:
            all_layers = [self.model.get_layer(index=layer_dict[lay]).output if lay in layer_real
                          else self.model.get_layer(index=layer_dict[-2]).output  # 如果给出不正确，就默认输出最后一层
                          for lay in self.layer_indexes]
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            encoder_layer = tf.keras.layers.Add(name="layer_add_albert")(all_layers_select)
        output = NonMaskingLayer(name="layer_non_masking_layer")(encoder_layer)
        self.output = output
        # self.output = [encoder_layer]
        self.input = self.model.inputs
        self.model = tf.keras.Model(self.input, self.output)

        # reader tokenizer
        self.token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                self.token_dict[token] = len(self.token_dict)
        self.vocab_size = len(self.token_dict)
        self.tokenizer = keras_bert.Tokenizer(self.token_dict)

    def sentence2idx(self, text, second_text=None):
        # text = extract_chinese(str(text).upper())
        input_id, input_type_id = self.tokenizer.encode(first=text, second=second_text, max_len=self.len_max)
        input_mask = len([1 for ids in input_id if ids ==1])
        return [input_id, input_type_id, input_mask]
        # return [input_id, input_type_id]
