# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/1/14 17:05
# @author  : Mo
# @function:


tags_res = ['m', 'vn', 'v', 'Yg', 'Tg', 'l', 'p', 'nt', 'y', 'Rg', 'e', 'i', 'an', 'q', 'k', 'nr', 'Ag', 'n', 'vvn', 'd', 'f', 'ad', 'vd', 'z', 'Mg', 'nx', 'a', 'h', 's', 'u', 'na', 'Bg', 'j', 'w', 'Ng', 'o', 'nz', 'ns', 'b', 'Vg', 'Dg', 'r', 't', 'c']
# ['Rg', 'nt', 'Ng', 'm', 'u', 'nx', 'an', 'na', 'b', 'd', 'c', 'vd', 'j', 'ns', 'ad', 's', 'z', 'Mg', 'vn', 'l', 't', 'f', 'v', 'vvn', 'n', 'r', 'Tg', 'Dg', 'Bg', 'i', 'nr', 'k', 'q', 'o', 'a', 'w', 'e', 'h', 'p', 'y', 'nz', 'Ag', 'Yg', 'Vg']

tags_res = [tr.upper() for tr in tags_res]

from macropodus.preprocess.tools_common import txt_read

tag_jiagus = txt_read("data/tag_jiagu.txt")
tag_jiebas = txt_read("data/tag_jieba.txt")

tgu = []
for tag_jiagu in tag_jiagus:
    tags = tag_jiagu.split("\u3000")
    tag = tags[0].strip()
    tgu.append(tag.upper())

tga = []
for tag_jieba in tag_jiebas:
    tags = tag_jieba.split("\t")
    tag = tags[0].strip()
    tga.append(tag.upper())

tgus = []
tgas = []
for tr in tags_res:
    if tr.upper() not in tgu:
        tgus.append(tr.upper())
    if tr.upper() not in tga:
        tgas.append(tr.upper())

tgus.sort()
tgas.sort()
print("jiagu: ")
print(tgus)
print("jieba: ")
print(tgas)

bbc = ['AG', 'B', 'BG', 'DG', 'E', 'H', 'I', 'J', 'K', 'L', 'MG', 'NA', 'NG', 'NX', 'O', 'RG', 'TG', 'VG', 'VVN', 'Y', 'YG', 'Z']
gg = 0
