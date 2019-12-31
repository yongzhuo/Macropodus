# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/18 22:10
# @author  : Mo
# @function: text summarize


# text_summarize of extractive
from macropodus.summarize.feature_base.word_significance import WordSignificanceSum
from macropodus.summarize.feature_base.text_pronouns import TextPronounsSum
from macropodus.summarize.graph_base.textrank import TextRankSum, TextRankKey
from macropodus.summarize.feature_base.text_teaser import TextTeaserSum
from macropodus.summarize.feature_base.mmr import MMRSum

from macropodus.summarize.topic_base.topic_lda import LDASum
from macropodus.summarize.topic_base.topic_lsi import LSISum
from macropodus.summarize.topic_base.topic_nmf import NMFSum

from macropodus.summarize.nous_base.lead_3 import Lead3Sum


# feature
wss = WordSignificanceSum()
tps = TextPronounsSum()
tts = TextTeaserSum()
mms = MMRSum()

# graph-3
trs = TextRankSum()
trk = TextRankKey()

# nous
l3s = Lead3Sum()

# topic
lds = LDASum()
lss = LSISum()
nms = NMFSum()

# summarization
text_pronouns = tps.summarize
text_teaser = tts.summarize
word_sign = wss.summarize
textrank = trs.summarize
lead3 = l3s.summarize
mmr = mms.summarize
lda = lds.summarize
lsi = lss.summarize
nmf = nms.summarize

# keyword
keyword = trk.keyword


def summarization(text, num=320, type_summarize="lda", topic_min=6, judge_topic=False, alpha=0.6, type_l='mix', model_type="textrank_sklearn", title=None):
    """
        文本摘要汇总
    :param text: str, like "你是。大漠帝国。不是吧错了。哈哈。我的。"
    :param num: int, like 32
    :param type_summarize: str, like "lda", must in ['text_pronouns',  'text_teaser', 'word_sign', 'textrank', 'lead3', 'mmr', 'lda', 'lsi', 'nmf']
    :return: 
    """

    if type_summarize=="text_pronouns": # title, str, 可填标题, like "震惊,MacropodusXXX"
        res = text_pronouns(text, num, title)
    elif type_summarize=="text_teaser": # title, str, 可填标题, like "震惊,MacropodusXXX"
        res = text_teaser(text, num, title)
    elif type_summarize=="word_sign": #
        res = word_sign(text, num)
    elif type_summarize=="textrank": # model_type 可填 'textrank_textrank4zh', 'text_rank_sklearn' or 'textrank_gensim'
        res = textrank(text, num)
    elif type_summarize=="lead3":
        res = lead3(text, num, type_l) # type_l 可填 'begin', 'end' or 'mix'
    elif type_summarize=="mmr":
        res = mmr(text, num, alpha) # alpha 可填 0-1
    elif type_summarize=="lda": # topic_min>1, judge_topic=True or False
        res = lda(text, num, topic_min, judge_topic)
    elif type_summarize=="lsi": # topic_min>1, judge_topic=True or False
        res = lsi(text, num, topic_min, judge_topic)
    elif type_summarize=="nmf": # topic_min>1, judge_topic=True or False
        res = nmf(text, num, topic_min, judge_topic)
    else:
        raise RuntimeError("your input type_summarize is wrong, it must be in "
                           "['text_pronouns',  'text_teaser', 'word_sign', "
                           "'textrank', 'lead3', 'mmr', 'lda', 'lsi', 'nmf']")
    return res
