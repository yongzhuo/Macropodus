# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/5 22:02
# @author  : Mo
# @function: clear text


def is_total_num(text):
    """
      判断是否是数字的
    :param text: str
    :return: boolean, True or false
    """
    try:
        text_clear = text.replace(" ", "").strip()
        number = 0
        for one in text_clear:
            if one.isdigit():
                number += 1
        if number == len(text_clear):
            return True
        else:
            return False
    except:
        return False

