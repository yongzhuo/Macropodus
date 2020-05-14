# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/16 22:11
# @author  : Mo
# @function: train bilstm-crf-?embedding


# 适配Linux和GPU
import pathlib
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_KERAS'] = '1'
project_path = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.parent.parent)
print(project_path)
sys.path.append(project_path)
# 模型图
from macropodus.network.graph.bilstm_crf import BilstmCRFGraph as Graph
# 数据预处理, 删除文件目录下文件
from macropodus.preprocess.tools_common import delete_file
# 地址
from macropodus.conf.path_config import path_model_dir, \
                                        path_seg_pku_1998_train, path_tag_people_1998_train, \
                                        path_ner_people_1998_train, path_ner_people_1998_valid, \
                                        path_tag_people_2014_train, path_tag_people_2014_valid, \
                                        path_ccks_2020, path_ccks_2020_ner
from macropodus.network.preprocess.preprocess_generator import PreprocessGenerator
# 计算时间
import time


def train_w2v(hyper_parameters=None, rate=1.0):
    if not hyper_parameters:
        hyper_parameters = {
            'len_max': 156,  # 句子最大长度, 固定推荐20-50, bert越长会越慢, 占用空间也会变大, 本地win10-4G设为20就好, 过大小心OOM
            'embed_size': 300,  # 768,  # 字/词向量维度, bert取768, word取300, char可以更小些
            'vocab_size': 20000,  # 这里随便填的，会根据代码里修改
            'trainable': True,  # embedding是静态的还是动态的, 即控制可不可以微调
            'level_type': 'char',  # 级别, 最小单元, 字/词, 填 'char' or 'word', 注意:word2vec模式下训练语料要首先切好
            'embedding_type': 'random',  # 级别, 嵌入类型, 还可以填'random'、 'bert'、 'albert' or 'word2vec"
            'model': {'label': 13,  # 类别数
                      'batch_size': 32,  # 批处理尺寸, 感觉原则上越大越好,尤其是样本不均衡的时候, batch_size设置影响比较大
                      'dropout': 0.5,  # 随机失活, 概率
                      'decay_step': 10000,  # 学习率衰减step, 每N个step衰减一次
                      'decay_rate': 0.999,  # 学习率衰减系数, 乘法
                      'epochs': 32,  # 训练最大轮次
                      'patience': 3,  # 早停,2-3就好            'gpu_memory_fraction': 0.76,  # gpu使用率
                      'lr': 1e-3,  # 学习率, bert取5e-5,其他取1e-3,如果acc较低或者一直不变,优先调这个, 对训练会有比较大的影响, 如果准确率一直上不去,可以考虑调这个参数
                      'l2': 0.5,  # l2正则化
                      'activate_rnn': 'tanh',  # rnn-layer中的激活函数, 即RNN激活函数, 可填'tanh', 'relu', 'signmoid'
                      'activate_classify': 'softmax',  # 最后一个layer, 即分类激活函数, 'softmax', 'signmoid'
                      'loss': 'categorical_crossentropy', # 损失函数, mse, categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy等
                      'metrics': 'accuracy',  # 保存更好模型的评价标准, accuracy, binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, sparse_top_k_categorical_accuracy
                      'optimizer_name': 'ADAM',  # 可填'ADAM', 'RADAM', 'RADAM,LOOKAHEAD'
                      'is_training': True,  # 训练后者是测试模型, 训练时候是True, 测试时候是False
                      'path_model_dir': os.path.join(path_model_dir, "ccks_2020_ner_RandomBiLstmCrf"),  # 保存当前训练模型的根目录
                      'model_path': os.path.join(path_model_dir, "bilstm_crf.model"),
                      # 模型地址, loss降低则保存的依据, save_best_only=True, save_weights_only=True
                      'path_hyper_parameters': os.path.join(path_model_dir, "hyper_parameters.json"),
                      # 模型(包括embedding)，超参数地址,
                      'path_fineture': os.path.join(path_model_dir, "embedding.model"),
                      # embedding trainable地址, 例如字向量、词向量、bert向量等
                      'path_l2i_i2l': os.path.join(path_model_dir, "l2i_i2l.json"),  # 类别转类标的字典
                      'num_rnn_layers': 1,  # rnn层数, 1, 2 or 3等
                      'rnn_type': 'LSTM',  # rnn类型,可以填"LSTM","GRU","CuDNNLSTM","CuDNNGRU"
                      'rnn_units': 256,  # rnn隐藏元, 128, 256, 512, 768, 1024等
                      'crf_mode': 'reg',  # crf类型, 可填'other', 'reg', 'pad'(包括句子实际长度)
                      },
            'embedding': {'layer_indexes': [-1, -2, -3, -4],  # bert取的层数
                          # 'corpus_path': '',     # embedding预训练数据地址,不配则会默认取conf里边默认的地址, keras-bert可以加载谷歌版bert,百度版ernie(需转换，https://github.com/ArthurRizar/tensorflow_ernie),哈工大版bert-wwm(tf框架，https://github.com/ymcui/Chinese-BERT-wwm)
                          },
            'data': {'train_data': path_ccks_2020_ner, # path_seg_pku_1998_train, # path_ner_people_1998_train, # path_tag_people_2014_train, # path_seg_pku_1998_train,  # path_ner_people_1998_train, # 训练数据
                     'val_data': path_ccks_2020_ner # path_tag_people_1998_train # path_seg_pku_1998_train # path_ner_people_1998_valid # path_tag_people_2014_valid # path_seg_pku_1998_train  # path_ner_people_1998_valid    # 验证数据
                     },
        }

    time_start = time.time()
    if not os.path.exists(hyper_parameters['model']['path_model_dir']):
        os.mkdir(hyper_parameters['model']['path_model_dir'])
    # 删除先前存在的模型和embedding微调模型等
    delete_file(hyper_parameters['model']['path_model_dir'])
    # 数据预处理初始化
    pg = PreprocessGenerator(os.path.join(hyper_parameters['model']['path_model_dir'], "l2i_i2l.json"))
    label_sets, _ = pg.preprocess_label2set(hyper_parameters['data']['train_data'], hyper_parameters['embedding_type'], encoding="utf-8")
    # 训练数据中试集序列类别个数
    hyper_parameters['model']['label'] = len(label_sets)
    # graph初始化
    graph = Graph(hyper_parameters)
    print("graph init ok!")
    ra_ed = graph.word_embedding
    # # 数据预处理, fit
    # x_train, y_train = pg.preprocess_label_question_to_idx_fit(embedding_type=hyper_parameters['embedding_type'],
    #                                                            path=hyper_parameters['data']['train_data'],
    #                                                            embed=ra_ed,
    #                                                            rate=rate,
    #                                                            crf_mode=hyper_parameters['model']['crf_mode'])
    # print("x_train init ok!")
    # x_val, y_val = pg.preprocess_label_question_to_idx_fit(embedding_type=hyper_parameters['embedding_type'],
    #                                                        path=hyper_parameters['data']['val_data'],
    #                                                        embed=ra_ed,
    #                                                        rate=rate,
    #                                                        crf_mode=hyper_parameters['model']['crf_mode'])
    # print("x_val init ok!")
    # # 训练
    # graph.fit(x_train, y_train, x_train, y_train,) # x_val, y_val)
    # fit_generator
    graph.fit_generator(ra_ed, rate, encoding="utf-8")

    # graph.fit_generator_kfold(ra_ed, rate, encoding="utf-8")
    print("耗时:" + str(time.time() - time_start))


def train_albert(hyper_parameters=None, rate=1.0):
    if not hyper_parameters:
        hyper_parameters = {
            'len_max': 334,  # 句子最大长度, 固定推荐20-50, bert越长会越慢, 占用空间也会变大, 本地win10-4G设为20就好, 过大小心OOM
            'embed_size': 768,  # 768,  # 字/词向量维度, bert取768, word取300, char可以更小些
            'vocab_size': 20000,  # 这里随便填的，会根据代码里修改
            'trainable': False,  # embedding是静态的还是动态的, 即控制可不可以微调
            'level_type': 'char',  # 级别, 最小单元, 字/词, 填 'char' or 'word', 注意:word2vec模式下训练语料要首先切好
            'embedding_type': 'albert',  # 级别, 嵌入类型, 还可以填'random'、 'bert'、 'albert' or 'word2vec"
            'model': {'label': 9,  # 类别数
                      'batch_size': 2,  # 批处理尺寸, 感觉原则上越大越好,尤其是样本不均衡的时候, batch_size设置影响比较大
                      'dropout': 0.5,  # 随机失活, 概率, keras和tf.keras上正好相反
                      'decay_step': 10000,  # 学习率衰减step, 每N个step衰减一次, 设置过小快速衰减会产生问题
                      'decay_rate': 0.999,  # 学习率衰减系数, 乘法, 最好是0.95以上吧
                      'epochs': 5,  # 训练最大轮次, ner比较小效果就比较好了
                      'patience': 1,  # 早停,2-3就好
                      'gpu_memory_fraction': 0.76,  # gpu使用率
                      'lr': 5e-5,  # 学习率, bert取5e-5,其他取1e-3,如果acc较低或者一直不变,优先调这个, 对训练会有比较大的影响, 如果准确率一直上不去,可以考虑调这个参数
                      'l2': 0.5,  # l2正则化
                      'activate_rnn': 'tanh',  # rnn-layer中的激活函数, 即RNN激活函数, 可填'tanh', 'relu', 'signmoid'
                      'activate_classify': 'softmax',  # 最后一个layer, 即分类激活函数, 'softmax', 'signmoid'
                      'loss': 'categorical_crossentropy', # 损失函数, mse, categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy等
                      'metrics': 'accuracy',  # 保存更好模型的评价标准, accuracy, binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, sparse_top_k_categorical_accuracy
                      'optimizer_name': 'ADAM',  # 可填'ADAM', 'RADAM', 'RADAM,LOOKAHEAD'
                      'is_training': True,  # 训练后者是测试模型, 训练时候是True, 测试时候是False
                      'path_model_dir': os.path.join(path_model_dir, "tag_people_2014_radom_bilstm_crf"),  # 保存当前训练模型的根目录
                      'model_path': os.path.join(path_model_dir, "bilstm_crf.model"),
                      # 模型地址, loss降低则保存的依据, save_best_only=True, save_weights_only=True
                      'path_hyper_parameters': os.path.join(path_model_dir, "hyper_parameters.json"),
                      # 模型(包括embedding)，超参数地址,
                      'path_fineture': os.path.join(path_model_dir, "embedding.model"),
                      # embedding trainable地址, 例如字向量、词向量、bert向量等
                      'path_l2i_i2l': os.path.join(path_model_dir, "l2i_i2l.json"),  # 类别转类标的字典
                      'num_rnn_layers': 1,  # rnn层数, 1, 2 or 3等
                      'rnn_type': 'LSTM',  # rnn类型,可以填"LSTM","GRU","CuDNNLSTM","CuDNNGRU"
                      'rnn_units': 768,  # rnn隐藏元, 128, 256, 512, 768, 1024等
                      'crf_mode': 'reg',  # crf类型, 可填'other', 'reg', 'pad'(包括句子实际长度)
                      },
            'embedding': {'layer_indexes': [1, 2, 3, 4], # [-1, -2, -3, -4],  # bert取的层数
                          # 'corpus_path': '',     # embedding预训练数据地址,不配则会默认取conf里边默认的地址, keras-bert可以加载谷歌版bert,百度版ernie(需转换，https://github.com/ArthurRizar/tensorflow_ernie),哈工大版bert-wwm(tf框架，https://github.com/ymcui/Chinese-BERT-wwm)
                          },
            # 'data': {'train_data': path_tag_people_1998_train, # path_tag_people_2014_train, # path_seg_pku_1998_train,  # path_ner_people_1998_train,  # path_seg_pku_1998_train,  # path_ner_people_1998_train, # 训练数据
            #          'val_data':path_tag_people_1998_train # path_ner_people_1998_valid # path_tag_people_2014_valid # path_seg_pku_1998_train  #  path_ner_people_1998_valid # path_seg_pku_1998_train  # path_ner_people_1998_valid    # 验证数据
            #          },
            'data': {'train_data': os.path.join(path_ccks_2020, "train_all.json"),
                     # path_seg_pku_1998_train, # path_ner_people_1998_train, # path_tag_people_2014_train, # path_seg_pku_1998_train,  # path_ner_people_1998_train, # 训练数据
                     'val_data': os.path.join(path_ccks_2020, "val.json")
                     # path_tag_people_1998_train # path_seg_pku_1998_train # path_ner_people_1998_valid # path_tag_people_2014_valid # path_seg_pku_1998_train  # path_ner_people_1998_valid    # 验证数据
                     },
        }

    time_start = time.time()
    if not os.path.exists(hyper_parameters['model']['path_model_dir']):
        os.mkdir(hyper_parameters['model']['path_model_dir'])
    # 删除先前存在的模型和embedding微调模型等
    delete_file(hyper_parameters['model']['path_model_dir'])
    # 数据预处理初始化
    from macropodus.network.preprocess.preprocess_generator import PreprocessGenerator
    pg = PreprocessGenerator(os.path.join(hyper_parameters['model']['path_model_dir'], "l2i_i2l.json"))
    label_sets, _ = pg.preprocess_label2set(hyper_parameters['data']['train_data'], hyper_parameters['embedding_type'], encoding = "gbk")
    # 训练数据中试集序列类别个数
    hyper_parameters['model']['label'] = len(label_sets)
    # graph初始化
    graph = Graph(hyper_parameters)
    print("graph init ok!")
    ra_ed = graph.word_embedding
    # 数据预处理, fit
    x_train, y_train = pg.preprocess_label_question_to_idx_fit(embedding_type=hyper_parameters['embedding_type'],
                                                               path=hyper_parameters['data']['train_data'],
                                                               embed=ra_ed,
                                                               rate=rate,
                                                               crf_mode=hyper_parameters['model']['crf_mode'], encoding = "gbk")

    x_val, y_val = pg.preprocess_label_question_to_idx_fit(embedding_type=hyper_parameters['embedding_type'],
                                                           path=hyper_parameters['data']['val_data'],
                                                           embed=ra_ed,
                                                           rate=rate,
                                                           crf_mode=hyper_parameters['model']['crf_mode'], encoding = "gbk")
    # 训练
    graph.fit(x_train, y_train, x_val, y_val)
    # # fit_generator
    # graph.fit_generator(ra_ed, rate)
    print("耗时:" + str(time.time() - time_start))


if __name__ == "__main__":
    # train_albert(rate=1)
    train_w2v(rate=1)
