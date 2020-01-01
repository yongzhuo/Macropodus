# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/2 21:08
# @author  : Mo
# @function: preprocess of network


from tensorflow.python.keras.utils import to_categorical
from macropodus.preprocess.tools_common import load_json
from macropodus.preprocess.tools_common import save_json
import numpy as np
import json
import os


class PreprocessGenerator:
    """
        数据预处理, 输入为csv格式, [label,ques]
    """

    def __init__(self, path_model_l2i_i2l):
        self.path_model_l2i_i2l = path_model_l2i_i2l
        self.l2i_i2l = None
        if os.path.exists(self.path_model_l2i_i2l):
            self.l2i_i2l = load_json(self.path_model_l2i_i2l)

    def prereocess_idx2label(self, pred):
        """
            类标(idx)转类别(label)
        :param pred: 
        :return: 
        """
        if os.path.exists(self.path_model_l2i_i2l):
            pred_i2l = {}
            i2l = self.l2i_i2l['i2l']
            for i in range(len(pred)):
                pred_i2l[i2l[str(i)]] = pred[i]
            pred_i2l_rank = [sorted(pred_i2l.items(), key=lambda k: k[1], reverse=True)]
            return pred_i2l_rank
        else:
            raise RuntimeError("path_fast_text_model_label2index is None")

    def prereocess_label2idx(self, pred):
        """
            类别(label)转类标(idx)
        :param pred: 
        :return: 
        """
        if os.path.exists(self.path_model_l2i_i2l):
            pred_l2i = {}
            l2i = self.l2i_i2l['l2i']
            for i in range(len(pred)):
                pred_l2i[pred[i]] = l2i[pred[i]]
            pred_l2i_rank = [sorted(pred_l2i.items(), key=lambda k: k[1], reverse=True)]
            return pred_l2i_rank
        else:
            raise RuntimeError("path_fast_text_model_label2index is None")

    def preprocess_label2set(self, path):
        """
            统计label个数, 以及具体的存在
        :param path: str, like 'train.json'
        :return: 
        """
        # 首先获取label,set,即存在的具体类
        label_sets = set(["<PAD>"])
        len_all = 0
        file_csv = open(path, "r", encoding="utf-8")
        for line in file_csv:
            len_all += 1
            if line.strip():
                ques_label = json.loads(line.strip())
                label_org = ques_label["label"]
                label_sets = label_sets | set(label_org)

        file_csv.close()
        return label_sets, len_all

    def preprocess_label_question_to_idx_fit_generator(self, embedding_type, batch_size, path, embed, rate=1, crf_mode='reg'):
        """
            fit_generator用, 将句子, 类标转化为数字idx
        :param embedding_type: str, like 'albert'
        :param batch_size: int, like 64
        :param path: str, like 'train.json'
        :param embed: class, like embed
        :param rate: float, like 0.9
        :param crf_mode: str, like 'reg', 'pad'
        :return: yield
        """
        # 首先获取label,set,即存在的具体类
        label_set, len_all = self.preprocess_label2set(path)
        # 获取label转index字典等, 如果label2index存在则不转换了, dev验证集合的时候用
        if not os.path.exists(self.path_model_l2i_i2l):
            count = 0
            label2index = {}
            index2label = {}
            for label_one in label_set:
                label2index[label_one] = count
                index2label[count] = label_one
                count = count + 1
            l2i_i2l = {}
            l2i_i2l['l2i'] = label2index
            l2i_i2l['i2l'] = index2label
            save_json(l2i_i2l, self.path_model_l2i_i2l)
        else:
            l2i_i2l = load_json(self.path_model_l2i_i2l)

        # 读取数据的比例
        len_ql = int(rate * len_all)
        if len_ql <= 500:  # sample时候不生效,使得语料足够训练
            len_ql = len_all

        def process_line(line, embed, l2i_i2l):
            """
                对每一条数据操作，获取label和问句index
            :param line: 
            :param embed: 
            :param l2i_i2l: 
            :return: 
            """
            # 对每一条数据操作，对question和label进行padding
            ques_label = json.loads(line.strip())
            label_org = ques_label["label"]
            label_index = [l2i_i2l["l2i"][lr] for lr in label_org]
            # len_sequence = len(label_index)
            que_embed = embed.sentence2idx("".join(ques_label["question"]))
            # label padding
            if embedding_type in ['bert', 'albert']:
                # padding label
                len_leave = embed.len_max - len(label_index) - 2
                if len_leave >= 0:
                    label_index_leave = [l2i_i2l["l2i"]["<PAD>"]] + [li for li in label_index] + [
                        l2i_i2l["l2i"]["<PAD>"]] + [l2i_i2l["l2i"]["<PAD>"] for i in range(len_leave)]
                else:
                    label_index_leave = [l2i_i2l["l2i"]["<PAD>"]] + label_index[0:embed.len_max - 2] + [
                        l2i_i2l["l2i"]["<PAD>"]]
            else:
                # padding label
                len_leave = embed.len_max - len(label_index)  # -2
                if len_leave >= 0:
                    label_index_leave = [li for li in label_index] + [l2i_i2l["l2i"]["<PAD>"] for i in range(len_leave)]
                else:
                    label_index_leave = label_index[0:embed.len_max]
            # 转为one-hot
            label_res = to_categorical(label_index_leave, num_classes=len(l2i_i2l["l2i"]))
            return que_embed, label_res

        file_csv = open(path, "r", encoding="utf-8")
        cout_all_line = 0
        cnt = 0
        x, y = [], []
        for line in file_csv:
            # 跳出循环
            if len_ql < cout_all_line:
                break
            cout_all_line += 1
            if line.strip():
                # 一个json一个json处理
                # 备注:最好训练前先处理,使得ques长度小于等于len_max(word2vec), len_max-2(bert, albert)
                x_line, y_line = process_line(line, embed, l2i_i2l)
                x.append(x_line)
                y.append(y_line.tolist())
                cnt += 1
            # 使用fit_generator时候, 每个batch_size进行yield
            if cnt == batch_size:
                # 通过两种方式处理: 1.嵌入类型(bert, word2vec, random), 2.条件随机场(CRF:'pad', 'reg')类型
                if embedding_type in ['bert', 'albert']:
                    x_, y_ = np.array(x), np.array(y)
                    x_1 = np.array([x[0] for x in x_])
                    x_2 = np.array([x[1] for x in x_])
                    x_3 = np.array([x[2] for x in x_])
                    if crf_mode == 'pad':
                        x_all = [x_1, x_2, x_3]
                    elif crf_mode == 'reg':
                        x_all = [x_1, x_2]
                    else:
                        x_all = [x_1, x_2]
                else:
                    x_, y_ = np.array(x), np.array(y)
                    x_1 = np.array([x[0] for x in x_])
                    x_2 = np.array([x[1] for x in x_])
                    if crf_mode == 'pad':
                        x_all = [x_1, x_2]
                    elif crf_mode == 'reg':
                        x_all = [x_1]
                    else:
                        x_all = [x_1]

                cnt = 0
                yield (x_all, y_)
                x, y = [], []

    def preprocess_label_question_to_idx_fit(self, embedding_type, path, embed, rate=1, crf_mode='reg'):
        """
            fit用, 关键:对每一条数据操作，获取label和问句index              
        :param embedding_type: str, like 'albert'
        :param path: str, like 'train.json'
        :param embed: class, like embed
        :param rate: float, like 0.9
        :param crf_mode: str, like 'reg', 'pad'
        :return: np.array
        """
        # 首先获取label,set,即存在的具体类
        label_set, len_all = self.preprocess_label2set(path)
        # 获取label转index字典等, 如果label2index存在则不转换了, dev验证集合的时候用
        if not os.path.exists(self.path_model_l2i_i2l):
            count = 0
            label2index = {}
            index2label = {}
            for label_one in label_set:
                label2index[label_one] = count
                index2label[count] = label_one
                count = count + 1
            l2i_i2l = {}
            l2i_i2l['l2i'] = label2index
            l2i_i2l['i2l'] = index2label
            save_json(l2i_i2l, self.path_model_l2i_i2l)
        else:
            l2i_i2l = load_json(self.path_model_l2i_i2l)

        # 读取数据的比例
        len_ql = int(rate * len_all)
        if len_ql <= 500:  # sample时候不生效,使得语料足够训练
            len_ql = len_all

        def process_line(line, embed, l2i_i2l):
            """
                对每一条数据操作，获取label和问句index
            :param line: 
            :param embed: 
            :param l2i_i2l: 
            :return: 
            """
            # 对每一条数据操作，对question和label进行padding
            ques_label = json.loads(line.strip())
            label_org = ques_label["label"]
            label_index = [l2i_i2l["l2i"][lr] for lr in label_org]
            # len_sequence = len(label_index)
            que_embed = embed.sentence2idx("".join(ques_label["question"]))
            # label padding
            if embedding_type in ['bert', 'albert']:
                # padding label
                len_leave = embed.len_max - len(label_index) - 2
                if len_leave >= 0:
                    label_index_leave = [l2i_i2l["l2i"]["<PAD>"]] + [li for li in label_index] + [
                        l2i_i2l["l2i"]["<PAD>"]] + [l2i_i2l["l2i"]["<PAD>"] for i in range(len_leave)]
                else:
                    label_index_leave = [l2i_i2l["l2i"]["<PAD>"]] + label_index[0:embed.len_max - 2] + [
                        l2i_i2l["l2i"]["<PAD>"]]
            else:
                # padding label
                len_leave = embed.len_max - len(label_index)  # -2
                if len_leave >= 0:
                    label_index_leave = [li for li in label_index] + [l2i_i2l["l2i"]["<PAD>"] for i in range(len_leave)]
                else:
                    label_index_leave = label_index[0:embed.len_max]
            # 转为one-hot
            label_res = to_categorical(label_index_leave, num_classes=len(l2i_i2l["l2i"]))
            return que_embed, label_res

        file_csv = open(path, "r", encoding="utf-8")
        cout_all_line = 0
        cnt = 0
        x, y = [], []
        for line in file_csv:
            # 跳出循环
            if len_ql < cout_all_line:
                break
            cout_all_line += 1
            if line.strip():
                # 一个json一个json处理
                # 备注:最好训练前先处理,使得ques长度小于等于len_max(word2vec), len_max-2(bert, albert)
                x_line, y_line = process_line(line, embed, l2i_i2l)
                x.append(x_line)
                y.append(y_line.tolist())
                cnt += 1

        # 通过两种方式处理: 1.嵌入类型(bert, word2vec, random), 2.条件随机场(CRF:'pad', 'reg')类型
        if embedding_type in ['bert', 'albert']:
            x_, y_ = np.array(x), np.array(y)
            x_1 = np.array([x[0] for x in x_])
            x_2 = np.array([x[1] for x in x_])
            x_3 = np.array([x[2] for x in x_])
            if crf_mode == 'pad':
                x_all = [x_1, x_2, x_3]
            elif crf_mode == 'reg':
                x_all = [x_1, x_2]
            else:
                x_all = [x_1, x_2]
        else:
            x_, y_ = np.array(x), np.array(y)
            x_1 = np.array([x[0] for x in x_])
            x_2 = np.array([x[1] for x in x_])
            if crf_mode == 'pad':
                x_all = [x_1, x_2]
            elif crf_mode == 'reg':
                x_all = x_1
            else:
                x_all = x_1
                # 使用fit的时候, return返回
        return x_all, y_
