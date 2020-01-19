# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/18 23:59
# @author  : Mo
# @function: path of macropodus


import sys
import os
path_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(path_root)


# path of basic of segnment
path_dict_macropodus = os.path.join(path_root, "data/dict/macropodus.dict")
path_dict_user = os.path.join(path_root, "data/dict/user.dict")
path_log_basic = os.path.join(path_root, "logs")

# path of cache
path_macropodus_w2v_char_cache = os.path.join(path_root, 'data/cache/word2vec_char.cache')
path_macropodus_dict_freq_cache = os.path.join(path_root, 'data/cache/macropodus.cache')

# path of basic of tookit
path_dict_pinyin = os.path.join(path_root, "data/dict/pinyin.dict")
path_dict_zh2han = os.path.join(path_root, "data/dict/zh2han.dict")

# path of embedding
path_embedding_word2vec_char = os.path.join(path_root, 'data/embedding/word2vec/w2v_model_wiki_char.vec')
path_embedding_bert = os.path.join(path_root, 'data/embedding/chinese_L-12_H-768_A-12/')
path_embedding_random_char = os.path.join(path_root, 'data/embedding/term_char.txt')
path_embedding_random_word = os.path.join(path_root, 'data/embedding/term_word.txt')
path_embedding_albert = os.path.join(path_root, 'data/embedding/albert_base_zh')

# path of train data of ner people 1998
path_ner_people_1998_train = os.path.join(path_root, "data/corpus/ner_people_1998/train.json")
path_ner_people_1998_valid = os.path.join(path_root, "data/corpus/ner_people_1998/dev.json")
# path of train data of seg pku 1998
path_seg_pku_1998_train = os.path.join(path_root, "data/corpus/seg_pku_1998/train.json")
# path of train data of tag people 1998
path_tag_people_1998_train = os.path.join(path_root, "data/corpus/tag_people_1998/train.json")
# path of train data of tag people 2014
path_tag_people_2014_train = os.path.join(path_root, "data/corpus/tag_people_2014/train.json")
path_tag_people_2014_valid = os.path.join(path_root, "data/corpus/tag_people_2014/dev.json")

# path of training model save dir
path_model_dir = os.path.join(path_root, "data", "model")
path_hyper_parameters = os.path.join(path_model_dir, "params.json")
path_model_l2i_i2l = os.path.join(path_model_dir, "l2i_i2l.json")
path_fineture = os.path.join(path_model_dir, "embedding.h5")
path_model = os.path.join(path_model_dir, "model.h5")
