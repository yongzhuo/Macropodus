# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/17 14:50
# @author  : Mo
# @function: similarity of sentence of word2vec


from macropodus.base.word2vec import W2v


class SimW2vChar(W2v):
    def __init__(self, use_cache=True):
        super().__init__(use_cache)

    def encode(self, sent, type_encode="other"):
        """
            生成句向量, 字符级别, char
        :param sent: str, like "大漠帝国"
        :param type_encode: str, like "avg", "other"
        :return: vector
        """
        sentence_vec = self.w2v_char.wv[self.w2v_char.index2word[1]] * 0
        len_sent = len(sent)
        for i in range(len_sent):
            word = sent[i]
            try:
                sentence_vec = sentence_vec + self.w2v_char.wv[word]
            except Exception as e:
                sentence_vec = sentence_vec + 0.01  # unknow_know词加0.01
        if type_encode == "avg":
            sentence_vec = sentence_vec / len_sent
        return sentence_vec

    def similarity(self, sent1, sent2, type_sim="total", type_encode="avg"):
        """
            相似度计算, 默认余弦相似度+jaccard相似度
        :param sen1: str, like "大漠帝国"
        :param sen2: str, like "Macropodus"
        :param type_sim: str, like "total" or "cosine"
        :param type_encode: str, like "other" or "avg"
        :return: float, like 0.998
        """
        if sent1 and sent2:
            encode_sen1 = self.encode(sent1, type_encode)
            encode_sen2 = self.encode(sent2, type_encode)
            score_res = self.cosine(encode_sen1, encode_sen2)
        else:
            score_res = 0.0
        if type_sim=="total":
            score_jaccard = self.jaccard(sent1, sent2)
            score_res = (score_res + score_jaccard)/2
        return score_res


if __name__ == '__main__':

    sent1 = "大漠帝国"
    sent2 = "macropodus"
    swc = SimW2vChar(use_cache=True)
    sen_encede = swc.encode(sent1)
    score = swc.similarity(sent1, sent2)
    print(score)
    gg = 0
    while True:
        print("请输入sent1:")
        sent1 = input()
        print("请输入sent2:")
        sent2 = input()
        print(swc.similarity(sent1, sent2))
