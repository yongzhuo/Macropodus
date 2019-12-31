# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/11/21 20:22
# @author   :Mo
# @function :an ai text calcultor of xiaomo


from macropodus.tookit.calculator_sihui.calcultor_function import rackets_replace, reagan, power, logarithm, fraction, factorial, fractiontwo
from macropodus.tookit.calculator_sihui.calcultor_number import extract_number, sph
from macropodus.tookit.calculator_sihui.calcultor_formula import result_formula
from macropodus.conf.path_log import get_logger_root
import re


logger = get_logger_root()


def StringToCalculateZero(words=''):
    """
        混合运算去除非计算式等无用词
    :param words: str
    :return: str
    """
    wordsspot = words.replace("点", ".")
    wordsmark = wordsspot.replace("分之", "fenzhi")
    wordsin = wordsmark.replace("正切", "zheng切").replace("正弦", "zheng弦").replace("正割", "zheng割").replace("正矢", "zheng矢")
    wordsadd = wordsin.replace("加上", "+").replace("加到", "+").replace("加", "+").replace("＋", "+").replace("正", "+")
    wordsminus = wordsadd.replace("减去", "-").replace("减", "-").replace("－", "-").replace("负", "-")
    wordsmult = wordsminus.replace("阶乘", "jiecheng的").replace("乘上", "*").replace("乘以", "*").replace("乘于","*").replace("乘", "*").replace("×", "*")
    wordsdivis01 = wordsmult.replace("除去", "/").replace("除以", "/").replace("除于", "/").replace("除","/").replace("÷", "/")
    wordsdivis02 = wordsdivis01.replace("从", "").replace("再", "").replace("在", "").replace("然后", "").replace("直", "").replace("到", "")
    wordbrackets = wordsdivis02.replace("（", "(").replace("）", ")").replace("=", "").replace("=", "")
    formula = wordbrackets.replace("左括号", "(").replace("右括号", "(").replace("的和", "").replace("的差", "").replace("的商", "").replace("的积", "")
    myformula_1 = formula.replace("*-", "*(-1)*").replace("\\*\\+", "*").replace("\\/\\-", "/(-1)/")
    myformula_2 = myformula_1.replace(" ", "").replace("\\+\\-", "\\-").replace("\\+\\+", "\\+").replace("\\-\\+", "\\-").replace("\\-\\-", "\\+")
    myformula_2 = rackets_replace("(", myformula_2)
    myformula_2 = rackets_replace(")", myformula_2)

    return myformula_2


def StringToCalculateOne(words):
    """
        简单句总调用
        求乘方，阶乘，指数，根式，三角函数，对数，最大最小公约数公倍数    
    :param words: str
    :return: str
    """
    try:
        res_reagan = reagan(words, words)  # 报错或不执行返回原来的数据
        res_power = power(words, words)
        # aa22 = triangle(complex[i], complex[i])
        res_logarithm = logarithm(words, words)
        rees_factorial = factorial(words, words)
        res_fraction = fraction(words, words)
        if (res_reagan != words):
            goal = res_reagan
        elif (res_power != words):
            goal = res_power
        # elif (aa22 != complex[i]):
        #     goal = aa22
        elif (res_logarithm != words):
            goal = res_logarithm
        elif (rees_factorial != words):
            goal = rees_factorial
        elif (res_fraction != words):
            goal = res_fraction
        else:
            oldwords = words.replace("的", "")
            oldwords = extract_number(oldwords)[0]
            goal = oldwords
        return goal
    except Exception as e:
        logger.info(str(e))
        return words


def StringToCalculateTwo(sentence=''):
    """
        复杂算式, 总调用， 分步计算，先计算三角函数，指数，对数
        1.取出操作符与数据（注意--，++，-，+开头这种）
        2.计算中间的，比如说根号12，2的7次方这种
    :param sentence: 
    :return: 
    """
    try:
        if sentence[0] == '+' or sentence[0] == '-':
            sentence = '0' + sentence
        minus = 0
        operators = []
        complex = re.split("[+*/-]", sentence)
        for s in sentence:
            if ((s == '+' or s == '-' or s == '*' or s == '/') & minus != 0 & minus != 2):
                operators.append("" + s)
                minus = minus + 1
            else:
                minus = 1
        # complex.append(float(formula[prePos:].strip()))
        formula = ""
        for i in range(len(complex)):
            if "" == complex[i]:
                complex[i] = " "
                formula = formula + complex[i] + operators[i]
                continue
            res_reagan = reagan(complex[i], complex[i]) #报错或不执行返回原来的数据
            res_power = power(complex[i], complex[i])
            # aa22 = triangle(complex[i], complex[i])
            res_logarithm = logarithm(complex[i], complex[i])
            res_factorial = factorial(complex[i], complex[i])
            res_fraction = fraction(complex[i], complex[i])

            if (res_reagan != complex[i]):
                goal = res_reagan
            elif (res_power != complex[i]):
                goal = res_power
            # elif (aa22 != complex[i]):
            #     goal = aa22
            elif (res_logarithm != complex[i]):
                goal = res_logarithm
            elif (res_factorial != complex[i]):
                goal = res_factorial
            elif (res_fraction != complex[i]):
                goal = res_fraction
            elif "(" in complex[i] or ")" in complex[i]:
                goal = sph.numberTranslator(target=complex[i].replace("的", ""))
            else:
                oldwords = complex[i].replace("的", "")
                oldwords = extract_number(oldwords)[0]
                goal = oldwords
            if goal == 'illegal math': #非法抛出
               return 'illegal math'
            if (i < len(complex) - 1):
                rest = goal + operators[i]
            else:
                rest = goal
            formula = formula + rest
        myformula = formula.replace("*-", "*(-1)*").replace("*+", "*").replace("/-", "/(-1)/")
        formulalast = myformula.replace(" ", "").replace("+-", "-").replace("++", "+").replace("-+", "-").replace("--","+")
    except Exception as e:
        logger.info(str(e))
        return sentence

    return formulalast


class Calculator:
    def __init__(self):
        self.tookit = "calculator_sihui"

    def calculator_sihui(self, sentence = ''):
        """
            思慧计算器总调用接口
        :param sentence:str, 输入句子,TEXT 
        :return: 
        """
        # 运算符转换
        sentence_wise = StringToCalculateZero(sentence)
        if not sentence_wise:
            return sentence
        # 混合运算
        sentence_replace = StringToCalculateTwo(sentence_wise)
        if ('/0' in sentence_replace and '/0.' not in sentence_replace) or sentence_replace == 'illegal math':
            return 'illegal math'
        for char in sentence_replace:
            if char not in '+-*/().0123456789':
                return 'could not calculate'
        #
        result = result_formula(sentence_replace)
        return result


if __name__ == "__main__":
    cal = Calculator()
    equation_sample = [
            '',
            '2(3*4)6+4(4)4',
            '（1+2）=',
            '1+2等于几',
            '100+30',
            '111+90-9等于几',
            "23 + 13 * ((25+(-9-2-5-2*3-6/3-40*4/(2-3)/5+6*3) * (9-2*6/3 + 5 *3*9/9*5 +10 * 56/(-14) )) - (-4*3)/ (3+3*3) )",
            '1-2-3-4-5-6',
            '134+123*898*123456789212310',
            '4*5*6/6/5',
            '(1+(2+(3-(4*6/4/3)-1)-1)-3)+(6-7)',
            '1+1+1',
            '1*1*2',
            '1+2+3+4+6',
            '1/2+2*3*4*5/5/4-1',
            '1+2(12/13)',
            '1+2+3+4+5+6+7+8+9+10',
            '-1+1+2+(-2)',
            '((-3)*2+1/2-3*4/4) +1',
            'LG0',
            'LOG100',
            '以2为底4的对数',
            '根号一百二十三加上1',
            '1加2的根号',
            '根号1加2的和',
            '以2为底4的对数',
            '2的六次方',
            '2的3次幂',
            '二的一次方',
            '四的平方',
            '十一的立方',
            '开11的立方根',
            '开3的7次方根',
            '13的阶乘',
            '根号四',
            '1除以0',

            '负一加上100加上50000',
            '2的8次方减6',
            '根号5乘以90',
            '2的8次方减6',
            '1的平方加根号2',
            '30的阶乘加90',
            '二分之一加1/3',
            ''
        ]

    for es in equation_sample:
        print('ff算式: ' + es)
        print('思慧计算器结果: ' + str(cal.calculator_sihui(es)))