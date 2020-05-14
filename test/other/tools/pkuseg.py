# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/1/10 15:00
# @author  : Mo
# @function:


import pkuseg

ps = pkuseg.pkuseg()
res = ps.cut("帝国主义要把我们的地瓜分掉")
print(res)