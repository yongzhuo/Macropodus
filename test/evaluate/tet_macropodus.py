# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/17 21:13
# @author  : Mo
# @function: test macropodus


import time
time_start = time.time()
import macropodus
print('macropodus初始化耗时: ' + str(time.time()-time_start) + 's')

# import sys
# import os
# print(os.name)
# print(sys.platform)

# macropodus.load_user_dict(path_user="user.json", type_user="json")
macropodus.add_word(word="斗鱼属")
macropodus.add_word(word="斗鱼科")
macropodus.add_word(word="鲈形目")
macropodus.save_add_words(word_freqs={"喜斗":32, "护卵":64, "护幼":132})
macropodus.add_word(word="坑爹的平衡性基金")
macropodus.save_add_words(word_freqs={"BBC":132})

print(macropodus.cut("坑爹的平衡性基金啊,坑爹呀斗鱼属，Macropodus (Lacépède, 1801)，鲈形目斗鱼科的一属鱼类。"
                          "本属鱼类通称斗鱼。因喜斗而得名。分布于亚洲东南部。中国有2种，即叉尾斗鱼，分布于长江及以南各省；"
                          "叉尾斗鱼，分布于辽河到珠江流域。其喜栖居于小溪、河沟、池塘、稻田等缓流或静水中。"
                          "雄鱼好斗，产卵期集草成巢，雄鱼口吐粘液泡沫，雌鱼产卵其中，卵浮性，受精卵在泡沫内孵化。雄鱼尚有护卵和护幼现象。"
                          ))

sen_calculate = "23 + 13 * (25+(-9-2-5-2*3-6/3-40*4/(2-3)/5+6*3))加根号144你算得几多"
sen_chi2num = "三千零七十八亿三千零十五万零三百一十二点一九九四"
sen_num2chi = 1994.1994
sent1 = "PageRank算法简介"
sent2 = "百度百科如是介绍他的思想:PageRank通过网络浩瀚的超链接关系来确定一个页面的等级。"
summary = "PageRank算法简介。" \
           "是上世纪90年代末提出的一种计算网页权重的算法! " \
           "当时，互联网技术突飞猛进，各种网页网站爆炸式增长。 " \
           "业界急需一种相对比较准确的网页重要性计算方法。 " \
           "是人们能够从海量互联网世界中找出自己需要的信息。 " \
           "百度百科如是介绍他的思想:PageRank通过网络浩瀚的超链接关系来确定一个页面的等级。 " \
           "Google把从A页面到B页面的链接解释为A页面给B页面投票。 " \
           "Google根据投票来源甚至来源的来源，即链接到A页面的页面。 " \
           "和投票目标的等级来决定新的等级。简单的说， " \
           "一个高等级的页面可以使其他低等级页面的等级提升。 " \
           "具体说来就是，PageRank有两个基本思想，也可以说是假设。 " \
           "即数量假设：一个网页被越多的其他页面链接，就越重）。 " \
           "质量假设：一个网页越是被高质量的网页链接，就越重要。 " \
           "总的来说就是一句话，从全局角度考虑，获取重要的信。 "

# 分词
words = macropodus.cut(summary)
print(words)
new_words = macropodus.find(summary)
print(new_words)

# 新词发现(findword, 默认接口)
sents = macropodus.find(text=summary, freq_min=2, len_max=7, entropy_min=1.2, aggregation_min=0.5, use_avg=True)
print(sents)
# 摘要
sum = macropodus.summarize(summary)
print(sum)
keyword = macropodus.keyword(summary)
print(keyword)
# 相似度
sim = macropodus.sim(sent1, sent2, type_sim="cosine")

sent1 = "叉尾斗鱼"
sent2 = "中国斗鱼生性好斗,适应性强,能在恶劣的环境中生存"

# 文本相似度(similarity)
sents = macropodus.sim(sent1, sent2, type_sim="total", type_encode="avg")
print(sents)
sents = macropodus.sim(sent1, sent2, type_sim="cosine", type_encode="single")
print(sents)
print(sim)
# tookit
score_calcul = macropodus.calculate(sen_calculate)
print(score_calcul)
res_chi2num = macropodus.chi2num(sen_chi2num)
print(res_chi2num)
res_num2chi = macropodus.num2chi(sen_num2chi)
print(res_num2chi)
