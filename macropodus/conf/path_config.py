# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/18 23:59
# @author  : Mo
# @function: path of macropodus


import sys
import os
path_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(path_root)


# path of basic
path_dict_macropodus = os.path.join(path_root, "data/dict/macropodus.dict")
path_dict_user = os.path.join(path_root, "data/dict/user.dict")
path_log_basic = os.path.join(path_root, "logs")

# path of cache
path_macropodus_w2v_char_cache = os.path.join(path_root, 'data/cache/word2vec_char.cache')
path_macropodus_dict_freq_cache = os.path.join(path_root, 'data/cache/macropodus.cache')

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

# path of training model save dir
path_model_dir = os.path.join(path_root, "data/model")
