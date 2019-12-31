# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/28 20:17
# @author  : Mo
# @function: basic of segment, dictionary


from macropodus.preprocess.tools_common import load_json, save_json, txt_read
from macropodus.conf.path_config import path_dict_macropodus, path_dict_user
from macropodus.conf.path_config import path_macropodus_dict_freq_cache
from macropodus.conf.path_log import get_logger_root
from collections import defaultdict
import pickle
import time
import os


logger = get_logger_root()


class SegBasic:
    def __init__(self, use_cache=True):
        # time_start = time.time()
        # 存在缓存则直接读取, 序列化加速缓存读取速度
        if use_cache and os.path.exists(path_macropodus_dict_freq_cache):
            with open(path_macropodus_dict_freq_cache, "rb") as fpmc:
                [self.dict_words_freq, self.num_words, self.dict_user] = pickle.load(fpmc)
                fpmc.close()
            # logger.info("seg: " + str(time.time()-time_start)) # 5.29, 5.26
        else:
            self.dict_words_freq = defaultdict()
            self.dict_user = {}
            self.load_macropodus_dict() # 默认字典
            self.load_user_dict() # 用户字典
            # logger.info("seg: " + str(time.time() - time_start)) # 10.13, 10.33
            # 第一次跑macropodus, 序列化需要的缓存
            if use_cache and not os.path.exists(path_macropodus_dict_freq_cache):
                with open(path_macropodus_dict_freq_cache, "wb") as fpmc:
                    pickle.dump([self.dict_words_freq, self.num_words, self.dict_user], fpmc)

    def load_macropodus_dict(self):
        """
            加载默认的基础字典
        :return: None
        """
        dict_macropodus = load_json(path_dict_macropodus)[0]  # (path_dict_jiagu)[0] # (path_dict_macropodus)[0] # 加载json字典文件
        dict_macropodus_def = defaultdict()  # 转为defaultdict
        for k,v in dict_macropodus.items():
            dict_macropodus_def[k] = v
        self.dict_words_freq = dict_macropodus_def  # {}词-词频字典

    def load_user_dict(self, path_user=path_dict_user, type_user="json"):
        """
            加载用户词典
        :param path_user:str, like '/home/user.dict' 
        :return: None
        """
        if not os.path.exists(path_user):
            raise RuntimeError("your path_user is not exist!")
        if type_user == "json":
            self.dict_user = load_json(path_user)[0]  # 加载json字典文件
            for k, v in self.dict_user.items():
                if k not in self.dict_words_freq:
                    self.dict_words_freq[k] = v   # 更新到总字典, words_freq
                else:
                    self.dict_words_freq[k] = self.dict_words_freq[k] + v   # 更新到总字典, words_freq
            self.num_words = sum(self.dict_words_freq.values())
        elif type_user == "txt":
            words_all = txt_read(path_user)
            for word_freq in words_all:
                wf = word_freq.split(" ") # 空格' '区分带不带词频的情况
                if len(wf) == 2:
                    word = wf[0]
                    freq = wf[1]
                else:
                    word = wf[0]
                    freq = 132
                if word not in self.dict_words_freq:
                    self.dict_words_freq[word] = freq   # 更新到总字典, words_freq
                else:
                    self.dict_words_freq[word] = self.dict_words_freq[word] + freq   # 更新到总字典, words_freq
            self.num_words = sum(self.dict_words_freq.values())
        elif type_user == "csv":
            words_all = txt_read(path_user)
            for word_freq in words_all:
                wf = word_freq.split(",") # 逗号','区分带不带词频的情况
                if len(wf)==2:
                    word = wf[0]
                    freq = wf[1]
                else:
                    word = wf[0]
                    freq = 132
                if word not in self.dict_words_freq:
                    self.dict_words_freq[word] = freq   # 更新到总字典, words_freq
                else:
                    self.dict_words_freq[word] = self.dict_words_freq[word] + freq   # 更新到总字典, words_freq
            self.num_words = sum(self.dict_words_freq.values())
        else:
            raise EOFError

    def add_word(self, word, freq=132):
        """
            新增词典到词语, 不可持久化, 重载消失
        :param word: str, like '大漠帝国'
        :param freq: int, like 132
        :return: None
        """
        assert type(word) == str
        if word in self.dict_words_freq:
            self.dict_words_freq[word] = self.dict_words_freq[word] if freq !=132 else freq
        else:
            self.dict_words_freq[word] = freq
        self.num_words += freq

    def delete_word(self, word):
        """
            删除词语, 不可持久化, 重载消失
        :param word_freqs: str, like '大漠帝国'
        :return: None
        """
        assert type(word) == str
        if word in self.dict_words_freq:
            self.num_words -= self.dict_words_freq[word]
            self.dict_words_freq.pop(word)

    def save_add_words(self, word_freqs):
        """
            新增词语到用户词典, 可持久化, 重载有效
        :param word_freqs: dict, like {'大漠帝国':132}
        :return: None
        """
        assert type(word_freqs) == dict
        for k, v in word_freqs.items():
            self.add_word(k, v)    # 新增到总字典, 不持久化
            self.dict_user[k] = v  # 新增到用户字典, 持久化
        save_json([self.dict_user], path_dict_user)

    def save_delete_words(self, words):
        """
            删除词语到用户词典, 可持久化, 重载有效
        :param word_freqs: list, like ['大漠帝国']
        :return: None
        """
        assert type(words) == list
        for w in words:
            self.delete_word(w) # 删除到总字典, 不持久化
            if w in self.dict_user: self.dict_user.pop(w) # 删除到用户字典, 持久化
        save_json([self.dict_user], path_dict_user)
