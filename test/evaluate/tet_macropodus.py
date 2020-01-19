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

# sent = "今日头条 白嫖 东风快递 令人喷饭 勿谓言之不预也 白嫖 口区 弓虽 口丕 我酸了 祖安人 迷惑行为 5G 996 007 1118 35 120 251 nmsl nsdd wdnmd CSGO 唱跳 rap 篮球 鸡你太美 cxk 盘它 撞梗 融梗 雨女无瓜 要你寡 刺激战场 绝地求生"
# sent = "狼灭 狼火 狼炎 狼焱 灵魂八问 硬核 奥力给 有内味了 awsl 影流之主 巨魔之王"
# words = sent.split(" ")
# word_dict = {}
# for w in words:
#     word_dict[w] = 132
# macropodus.save_add_words(word_freqs=word_dict)

print(macropodus.cut("坑爹的平衡性基金啊,坑爹呀斗鱼属，Macropodus (Lacépède, 1801)，鲈形目斗鱼科的一属鱼类。"
                          "本属鱼类通称斗鱼。因喜斗而得名。分布于亚洲东南部。中国有2种，即叉尾斗鱼，分布于长江及以南各省；"
                          "叉尾斗鱼，分布于辽河到珠江流域。其喜栖居于小溪、河沟、池塘、稻田等缓流或静水中。"
                          "雄鱼好斗，产卵期集草成巢，雄鱼口吐粘液泡沫，雌鱼产卵其中，卵浮性，受精卵在泡沫内孵化。雄鱼尚有护卵和护幼现象。"
                          ))

sen_calculate = "23 + 13 * (25+(-9-2-5-2*3-6/3-40*4/(2-3)/5+6*3))加根号144你算得几多"
sen_chi2num = "三千零七十八亿三千零十五万零三百一十二点一九九四"
sen_num2chi = 1994.1994
sen_roman2int = "IX"
sen_int2roman = 132
# sent1 = "PageRank算法简介"
# sent2 = "百度百科如是介绍他的思想:PageRank通过网络浩瀚的超链接关系来确定一个页面的等级。"
sent1 = "香蕉的翻译"
sent2 = "用英语说香蕉"
summary = "四川发文取缔全部不合规p2p。字节跳动与今日头条。成都日报，成都市，李太白与杜甫"\
           "PageRank算法简介。" \
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

# 分词(词典最大概率分词DAG)
words = macropodus.cut(summary)
print(words)
# 新词发现
new_words = macropodus.find(summary)
print(new_words)
# 文本摘要
sum = macropodus.summarize(summary)
print(sum)
# 关键词抽取
keyword = macropodus.keyword(summary)
print(keyword)
# 文本相似度
sim = macropodus.sim(sent1, sent2)
print(sim)
# tookit
# 计算器
score_calcul = macropodus.calculate(sen_calculate)
print(score_calcul)
# 中文数字与阿拉伯数字相互转化
res_chi2num = macropodus.chi2num(sen_chi2num)
print(res_chi2num)
res_num2chi = macropodus.num2chi(sen_num2chi)
print(res_num2chi)
# 阿拉伯数字与罗马数字相互转化
res_roman2int = macropodus.roman2num(sen_roman2int)
print(res_roman2int)
res_int2roman = macropodus.num2roman(sen_int2roman)
print(res_int2roman)
# 中文汉字转拼音
res_pinyin = macropodus.pinyin(summary)
print(res_pinyin)
# 中文繁简转化
res_zh2han = macropodus.zh2han(summary)
print(res_zh2han)
res_han2zh = macropodus.han2zh(res_zh2han)
print(res_han2zh)

# 命名实体提取albert+bilstm+crf, 需要安装tensorflow==1.15.0和下载模型
summary = ["美丽的广西是我国华南地区的一颗璀璨的明珠,山清水秀生态美,风生水起万象新。", "广西壮族自治区，简称“桂”，是中华人民共和国省级行政区"]
res_ner = macropodus.ner(summary[0])
print(res_ner)
res_ners = macropodus.ners(summary)
print(res_ners)
