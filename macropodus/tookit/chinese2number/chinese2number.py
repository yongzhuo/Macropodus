# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/11/17 22:17
# @author   :Mo
# @function :change chinese digit to Arab or reversed


from macropodus.preprocess.tools_clear import is_total_num
import random


# chinese_to_number, 单位-数字
unit_dict = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}
unit_dict_keys = unit_dict.keys()
digit_dict = {"零": 0, "一": 1, "二": 2, "两": 2, "俩": 2, "三": 3,
             "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

# number_to_chinese, 单位-数字
num_dict = { 0: "零", 1: "一", 2: "二", 3: "三", 4: "四",
             5: "五", 6: "六", 7: "七", 8: "八", 9: "九" }
unit_map = [ ["", "十", "百", "千"],       ["万", "十万", "百万", "千万"],
             ["亿", "十亿", "百亿", "千亿"], ["兆", "十兆", "百兆", "千兆"] ]
unit_step = ["万", "亿", "兆"]


class Chi2Num():
    def __init__(self):
        self.result = 0.0
        self.result_last = 0.0
        # 字符串分离
        self.str_billion = ""  # 亿
        self.str_billion_hundred = ""  # 亿万
        self.str_billion_one = ""
        self.str_thousand_ten = ""  # 万
        self.str_single = ""  # one

    def free_zero_and_split_three_parts(self, text):
        """
           去零切分成三部分
        :param text:str 
        :return: 
        """
        assert type(text) == str
        if "零" in text:
            text = text.replace("零", "")
        # 分切成三部分
        index = 0
        flag_billion = True # 亿
        flag_billion_hundred = True # 万亿
        flag_thousand_ten = True #万
        len_text = len(text)
        for i in range(len_text):
            if "亿" == text[i]:
                # 存在亿前面也有万的情况，小分节
                self.str_billion = text[0:i]
                if text.find("亿") > text.find("万"):
                    for j in range(len(self.str_billion)):
                        if "万" == self.str_billion[j]:
                            flag_billion_hundred = False
                            self.str_billion_hundred = self.str_billion[0:j]
                            self.str_billion_one = self.str_billion[j+1:]
                # 如何亿字节中没有万，直接赋值
                if flag_billion_hundred:
                    self.str_billion_one = self.str_billion
                index = i + 1
                flag_billion = False
                # 分节完毕
                self.str_single = text[i + 1:]
            if "万" == text[i] and text.find("亿") < text.find("万"):
                self.str_thousand_ten = text[index:i]
                self.str_single = text[i+1:]
                flag_thousand_ten = False
        if flag_billion and flag_thousand_ten:
            self.str_single = text

    def str_to_number(self, text):
        """
           string change to number
        :param text: str
        :return: 
        """
        assert type(text) == str
        number_res = 0
        number_1 = 0
        number_2 = 0
        number_3 = 0
        if not text:
            return 0
        len_text = len(text)
        for i in range(len_text):
            # 数字
            if text[i] in digit_dict:
                number_1 = digit_dict[text[i]]
                if i == len_text - 1:
                    number_res += number_1
            # 单位
            elif text[i] in unit_dict:
                number_2 = unit_dict[text[i]]
                if number_1==0 and number_2==10:
                    number_3 = number_2
                else:
                    number_3 = number_1 * number_2
                    # 清零避免重复读取
                    number_1 = 0
                number_res += number_3
            # 处理形如 "二点13亿", "1.56万" 这样的情况
            else:
                try:
                    text_else_str = [str(digit_dict[tet]) if tet in digit_dict else tet for tet in text]
                    number_res = float("".join(text_else_str))
                except:
                    number_res = 0
        return number_res

    def compose_integer(self, text):
        """
            整数转数字, 合并
        :param text:str, input of chinese, eg.["一百", "三千零七十八亿三千零十五万零三百一十二"] 
        :return: float, result of change chinese to digit
        """
        assert type(text) == str
        self.result = 0.0
        self.result_last = 0.0

        text = text.replace("兆", "万亿").replace("点", ".").strip(".").strip()
        len_text = len(text)
        # 判断十百千万在不在text里边，在的话就走第二个
        flag_pos = True
        for unit_dict_key in unit_dict_keys:
            if unit_dict_key in text:
                flag_pos = False
                break
        # 分三种情况，全数字返回原值，有中文unit_dict_keys就组合， 没有中文unit_dict_keys整合
        if is_total_num(text):
            digit_float = float(text)
            return digit_float
        elif flag_pos:
            result_pos = ""
            for i in range(len_text):
                if "."!=text[i] and not text[i].isdigit():
                    result_pos += str(digit_dict[text[i]])
                else:
                    result_pos += text[i]
            self.result_last = float(result_pos)
        else:
            self.free_zero_and_split_three_parts(text)
            float_billion_hundred = self.str_to_number(self.str_billion_hundred)
            float_billion_one = self.str_to_number(self.str_billion_one)
            float_thousand_ten = self.str_to_number(self.str_thousand_ten)
            float_single = self.str_to_number(self.str_single)

            self.result = float((float_billion_hundred * 10000 + float_billion_one) * 100000000 + float_thousand_ten * 10000 + float_single)
            self.result_last = self.result
            # 重置
            self.str_billion = ""  # 亿
            self.str_billion_hundred = ""  # 亿万
            self.str_billion_one = ""
            self.str_thousand_ten = ""  # 万
            self.str_single = ""  # one
        return self.result_last

    def compose_decimal(self, text):
        """
            中文小数转数字
        :param text:str, input of chinese, eg.["一百", "三千零七十八亿三千零十五万零三百一十二"] 
        :return: float, result of change chinese to digit
        """
        assert type(text) == str
        self.result = 0.0
        self.result_last = 0.0
        self.result_start = 0.0

        text = text.replace("兆", "万亿").replace("点", ".").strip()
        if "." in text:
            # 判断十百千万在不在.号后边，在的话就走compose_integer()，并且返回
            pos_point = text.find(".")
            for unit_dict_key in unit_dict_keys:
                if unit_dict_key in text:
                   if pos_point < text.find(unit_dict_key):
                       return  self.compose_integer(text)
            # 否则就是有小数
            texts = text.split(".")
            text_start = texts[0]
            text_end = texts[1]

            # 处理整数部分
            if "0"==text_start or "零"==text_start:
                self.result_start = "0."
            else:
                self.result_start = str(int(self.compose_integer(text_start))) + "."
            # 处理尾部，就是后边小数部分
            result_pos = ""
            len_text = len(text_end)
            for i in range(len_text):
                if "."!=text_end[i] and not text_end[i].isdigit():
                    result_pos += str(digit_dict[text_end[i]])
                else:
                    result_pos += text_end[i]
            # 拼接
            self.result_last = float(self.result_start + result_pos) if result_pos.isdigit() else self.result_start

        return self.result_last


class Num2Chi():
    """
       codes reference: https://github.com/tyong920/a2c
    """
    def __init__(self):
        self.result = ""

    def number_to_str(self, data):
        assert type(data) == float or int
        res = []
        count = 0
        # 倒转
        str_rev = reversed(str(data))  # seq -- 要转换的序列，可以是 tuple, string, list 或 range。返回一个反转的迭代器。
        for i in str_rev:
            if i is not "0":
                count_cos = count // 4  # 行
                count_col = count % 4   # 列
                res.append(unit_map[count_cos][count_col])
                res.append(num_dict[int(i)])
                count += 1
            else:
                count += 1
                if not res:
                    res.append("零")
                elif res[-1] is not "零":
                    res.append("零")
        # 再次倒序，这次变为正序了
        res.reverse()
        # 去掉"一十零"这样整数的“零”
        if res[-1] is "零" and len(res) is not 1:
            res.pop()

        return "".join(res)

    def decimal_chinese(self, data):
        assert type(data) == float or int
        data_str = str(data)
        if "." not in data_str:
            res = self.number_to_str(data_str)
        else:
            data_str_split = data_str.split(".")
            if len(data_str_split) is 2:
                res_start = self.number_to_str(data_str_split[0])
                res_end = "".join([num_dict[int(number)] for number in data_str_split[1]])
                res = res_start + random.sample(["点", "."], 1)[0] + res_end
            else:
                res = str(data)
        return res


if __name__=="__main__":
    ######  1.测试阿拉伯数字转中文  ######################################################
    ntc = Num2Chi()
    for i in range(1945, 2100):
        print(ntc.decimal_chinese(i))
    print(ntc.decimal_chinese(0.112354))
    print(ntc.decimal_chinese(1024.112354))



    ######  2.测试中文转阿拉伯   ########################################################
    ctn = Chi2Num()
    tet_base = [ "1.3", "一.12", "一", "一十五", "十五", "二十", "二十三", "一百","一百零一",
                 "一百一十", "一百一十一", "一千", "一千零一","一千零三十一",
                 "一万零二十一", "一万零三百二十一", "一万一千三百二十一", "三千零十五万",
                 "三千零一十五万", "三千五百六十八万零一百零一", "五十亿三千零七十五万零六百二十二",
                 "十三亿三千零十五万零三百一十二", "一千二百五十八亿","三千零十五万零三百一十二",
                 "一千二百五十八万亿", "三千三百二十一", "三百三十一", "二十一", "三百二十一",
                 "一千二百五十八亿零三千三百二十一", "两百", "两千两百二十二", "两亿两千万两百万两百二十二",
                 "三千零七十八亿三千零十五万零三百一十二"]

    tet_decimal = ["1.3", "三千零七十八亿三千零十五万零三百一十二", "二点13亿", "1.56万", "十八.12", "一十八点九一", "零点一", "零点123", "十八点一", "零", "一千零九十九点六六"]
    # 测试是小数
    for tet_d in tet_decimal:
        print(ctn.compose_decimal(tet_d))
    # 测试是整数
    for tet_b in tet_base:
        print(ctn.compose_integer(tet_b))

