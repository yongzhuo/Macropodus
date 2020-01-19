# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/1/2 9:14
# @author  : Mo
# @function: 罗马数字与阿拉伯数字相互转化


class RI:
    def __init__(self):
        self.algorithm = "roman2int"

    def roman2int(self, roman: str) -> int:
        """
            罗马数字转阿拉伯数字
        :param roman: str, like "IX"
        :return: int, like 9
        """
        roman2int_dict = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9,
                      'X': 10, 'XL': 40, 'L': 50, 'XC': 90,
                      'C': 100, 'CD': 400, 'D': 500, 'CM': 900,
                      'M': 1000}
        nums = 0
        while roman:
            if roman[0:2] in roman2int_dict.keys():
                nums += roman2int_dict[roman[0:2]]
                roman = roman[2:]
            elif roman[0:1] in roman2int_dict.keys():
                nums += roman2int_dict[roman[0:1]]
                roman = roman[1:]
        return nums

    def int2roman(self, num: int) -> str:
        """
            阿拉伯数字转罗马数字
        :param num: int, like 199
        :return: str, like "CXCIX"
        """
        int2roman_dict = {1: 'I', 4: 'IV', 5: 'V', 9: 'IX',
                          10: 'X', 40: 'XL', 50: 'L', 90: 'XC',
                          100: 'C', 400: 'CD', 500: 'D', 900: 'CM', 1000: 'M'}
        res = ""
        for key in sorted(int2roman_dict.keys())[::-1]:
            if (num == 0):
                break
            tmp = num // key
            if (tmp == 0):
                continue
            res += int2roman_dict[key] * (tmp)
            num -= key * (tmp)
        return res


if __name__ == '__main__':
    ri = RI()
    roman = "LVIII" # "IX" # "LVIII"
    num = 199
    res1 = ri.roman2int(roman)
    res2 = ri.int2roman(num)
    print(res1)
    print(res2)
