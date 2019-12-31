# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/21 23:11
# @author  : Mo
# @function:


from macropodus.preprocess.tools_common import load_json, save_json
from macropodus.preprocess.tools_common import txt_write, txt_read
import json

pku_training = txt_read("pku_training.utf8")
file = open("pku_train.json", "w", encoding="utf-8")
pku_ = []
for pku in pku_training:
    pkus = pku.split("  ")
    label_pkus = ""
    for pku_sig in pkus:
        len_pku = len(pku_sig)
        if len_pku==1:
            label_pkus += "S"
        elif len_pku==2:
            label_pkus += "BE"
        else:
            label_pkus += "B" + "M"*(len_pku-2) + "E"
    label_pkus_l = list(label_pkus)
    pku_res = {}
    pku_res["question"] = list("".join(pkus))
    pku_res["label"] = label_pkus_l
    p_json = json.dumps(pku_res, ensure_ascii=False)
    file.write(p_json + "\n")
#     pku_.append(pku_res)
# save_json(pku_, "pku_train.json")
