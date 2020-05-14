# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/5 20:23
# @author  : Mo
# @function: data utils of ml, text_summarization


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import macropodus
import re


__all__ = ["extract_chinese",
           "macropodus_cut",
           "jieba_tag_cut",
           "cut_sentence",
           "remove_urls",
           "tfidf_fit",
           "tfidf_sim"
           ]


def extract_chinese(text):
    """
      只提取出中文、字母和数字
    :param text: str, input of sentence
    :return: str
    """
    chinese_exttract = ''.join(re.findall(u"([\u4e00-\u9fa5A-Za-z0-9@. ])", text))
    return chinese_exttract


def jieba_tag_cut(text):
    """
        jieba cut and tagged
    :param text:str 
    :return: dict
    """
    import jieba.posseg as pseg
    words = pseg.cut(text)
    return dict(words)


def macropodus_cut(text):
    """
      Macropodus cut
    :param text: input sentence
    :return: list
    """
    return macropodus.cut(text)


def cut_sentence(text, use_type="summarize"):
    """
        分句(文本摘要)
    :param sentence:str, like "大漠帝国"
    :param use_type:str, like "summarize" or "new-word-discovery"
    :return:list
    """
    if use_type=="summarize":
        re_sen = re.compile('[:;!?。：；？！\n\r]') #.不加是因为不确定.是小数还是英文句号(中文省略号......)
    elif use_type=="new-word-discovery":
        re_sen = re.compile('[,，"“”、<>《》{}【】:;!?。：；？！\n\r]') #.不加是因为不确定.是小数还是英文句号(中文省略号......)
    else:
        raise RuntimeError("use_type must be 'summarize' or 'new-word-discovery'")
    sentences = re_sen.split(text)
    sen_cuts = []
    for sen in sentences:
        if sen and str(sen).strip():
            sen_cuts.append(sen)
    return sen_cuts


def remove_urls(text):
    """
        删除https/http等无用url
    :param text: str
    :return: str
    """
    text_remove_url = re.sub(r'(全文：)?(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                             '', text, flags=re.MULTILINE)
    return text_remove_url


def gram_uni_bi_tri(text):
    """
        获取文本的unigram, trugram, bigram等特征
    :param text: str
    :return: list
    """
    len_text = len(text)
    gram_uni = []
    gram_bi = []
    gram_tri = []
    for i in range(len_text):
        if i + 3 <= len_text:
            gram_uni.append(text[i])
            gram_bi.append(text[i:i+2])
            gram_tri.append(text[i:i+3])
        elif i + 2 <= len_text:
            gram_uni.append(text[i])
            gram_bi.append(text[i:i+2])
        elif i + 1 <= len_text:
            gram_uni.append(text[i])
        else:
            break
    return gram_uni, gram_bi, gram_tri


def get_ngrams(text, ns=[1], use_type="summarize", len_max=7):
    """
        获取文本的ngram等特征
    :param text: str, like "大漠帝国"
    :param ns: list, like [1, 2, 3]
    :param type: str, like "summarize" or "new-word-discovery"
    :param type: int, like 6, 7
    :return: list<list> or list
    """
    if type(ns) != list:
        raise RuntimeError("ns of function get_ngram() must be list!")
    for n in ns:
        if n < 1:
            raise RuntimeError("enum of ns must '>1'!")
    len_text = len(text)
    ngrams = []
    if use_type == "summarize": # 分别返回uni, bi, tri...
        for n in ns:
            ngram_n = []
            for i in range(len_text):
                if i + n <= len_text:
                    ngram_n.append(text[i:i + n])
                else:
                    break
            if not ngram_n:
                ngram_n.append(text)
            ngrams.append(ngram_n)
    else: # 只返回一个list
        for i in range(len_text):
            ngrams += [text[i: j + i]
                       for j in range(1, min(len_max + 1, len_text - i + 1))]
    return ngrams


def tfidf_fit(sentences):
    """
       tfidf相似度
    :param sentences: str
    :return: list, list, list
    """
    # tfidf计算
    model = TfidfVectorizer(ngram_range=(1, 2), # 3,5
                            stop_words=[' ', '\t', '\n'],  # 停用词
                            max_features=10000,
                            token_pattern=r"(?u)\b\w+\b",  # 过滤停用词
                            min_df=1,
                            max_df=0.9,
                            use_idf=1,  # 光滑
                            smooth_idf=1,  # 光滑
                            sublinear_tf=1, )  # 光滑
    matrix = model.fit_transform(sentences)
    return matrix


def tdidf_sim(sentences):
    """
       tfidf相似度
    :param sentences: 
    :return: 
    """
    # tfidf计算
    model = TfidfVectorizer(tokenizer=macropodus_cut,
                            ngram_range=(1, 2), # 3,5
                            stop_words=[' ', '\t', '\n'],  # 停用词
                            max_features=10000,
                            token_pattern=r"(?u)\b\w+\b",  # 过滤停用词
                            min_df=1,
                            max_df=0.9,
                            use_idf=1,  # 光滑
                            smooth_idf=1,  # 光滑
                            sublinear_tf=1, )  # 光滑
    matrix = model.fit_transform(sentences)
    matrix_norm = TfidfTransformer().fit_transform(matrix)
    return matrix_norm


if __name__ == '__main__':
    text = "你喜欢谁,小老弟,你好烦哇。"
    # gg = jieba_tag_cut("我不再喜欢你，正如你的不喜欢我")
    grams = get_ngrams(text, use_type="new-word-discovery", len_max=7)
    # print(gg)
    print(grams)
