# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/11/21 23:38
# @author   :Mo
# @function :calcultor of text, not filter and redundancy 


from macropodus.conf.path_log import get_logger_root
import re


logger = get_logger_root()


def change_symbol(formula):
    """
        提取负号
        eg：-9-2-5-2*3-5/3-40*4/-1.0/5+6*3  ===>  -(9+2+5+2*3+5/3+40*4/1.0/5-6*3)
    :param formula: 
    :return: 
    """
    def primary_change(for_str):  # 把算式中的全角 + - 对应换成 - +
        temp = for_str.split("+")
        new_formula = []
        for value in temp:
            value = value.replace("-", "+")
            new_formula.append(value)
        return "-".join(new_formula)

    if formula.startswith("-"):
        formula = formula.replace("-", "", 1)
        formula = primary_change(formula)
        formula = formula.join(["-(", ")"])
    elif formula.startswith("+"):
        formula = primary_change(formula)
        formula = formula.join(["-(", ")"])
    else:
        formula = primary_change(formula)
        formula = formula.join(["-(-", ")"])
    return formula


def remove_repeat(formula):
    """
        去掉连续的重复的运算符
    :param formula: str, like: "1++2"
    :return: str, like:"1+2"
    """
    temp = formula.replace("++", "+")
    temp = temp.replace("+-", "-")
    temp = temp.replace("-+", "-")
    temp = temp.replace("--", "+")
    temp = temp.replace("*+", "*")
    temp = temp.replace("+*", "*")
    temp = temp.replace("/+", "/")
    temp = temp.replace("+/", "/")
    return temp


def has_special_operator(formula, special_operator):
    """
        判断是否有 *+ +- /- 之类的运算符
    :param formula: 
    :param special_operator: 
    :return: 
    """
    for operator in special_operator:
        if formula.find(operator) != -1:
            return operator
    return ""


def handle_special_operator(formula, operator):
    """
        如果有 "*-", "-*", "/-", "-/" 这些运算符，
        提取负号，去掉重复的运算符
    :param formula: 
    :param operator: 
    :return: 
    """
    temp = ""
    regex = "\d*[.]?\d+"
    opera = operator.replace("*", "[*]")
    ret = re.compile(opera.join([regex, regex]))
    while ret.search(formula):
        search_res = ret.search(formula).group()
        if operator.find("*") != -1:
            temp = search_res.replace(operator, "*")
        elif operator.find("/") != -1:
            temp = search_res.replace(operator, "/")
        temp = "-".join(["", temp])
        formula = formula.replace(search_res, temp, 1)
    return formula


def has_parentheses(formula):
    """
        判断是否还有括号
    :param formula: str
    :return: boolean
    """
    if re.search("[()]", formula):
        return True
    return False


def judge_illegal(formula):
    """
        判断括号是否匹配完全，运算符是否合法
        没有考虑  **  //  的计算
    :param formula: str
    :return: str
    """
    if len(re.findall("[(]", formula)) != len(re.findall("[)]", formula)):
        return True
    if formula.startswith("*") or formula.startswith("/"):
        return True
    if has_special_operator(formula, ["*/", "/*", "**", "//"]):
        return True
    return False


def calculator_formula(formula):
    """
        计算算式，这里计算的是不带括号的算式
    计算次序是 / * - +
    计算过程中出现括号则停止计算，返回当前的算式
    :param formula: 
    :return: 
    """
    def primary_operator(for_str, operation):
        try:
            primary_result = 0
            regex = "\d*[.]?\d*"
            ret = re.compile(operation.join(["[", "]"]).join([regex, regex]))
            while ret.search(for_str):
                ret_opera = has_special_operator(for_str, ["*-", "-*", "/-", "-/"])
                while ret_opera:
                    for_str = handle_special_operator(for_str, ret_opera)
                    ret_opera = has_special_operator(for_str, ["*-", "-*", "/-", "-/"])
                while has_special_operator(for_str, ["+-", "-+", "++", "--", "+*", "*+", "+/", "/+"]):
                    for_str = remove_repeat(for_str)
                # print("primary_operator:", for_str)
                if has_parentheses(for_str):
                    return for_str
                if for_str.startswith("-"):
                    temp = re.findall("^-\d*[.]?\d*$", for_str)
                    if temp:
                        return temp[0]
                    return change_symbol(for_str)
                if for_str.startswith("+"):
                    for_str = for_str.replace("+", "", 1)
                if not ret.search(for_str):
                    continue
                search_res = ret.search(for_str).group()
                operand_list = search_res.split(operation)
                if operation == "/":
                    primary_result = float(operand_list[0]) / float(operand_list[1])
                elif operation == "*":
                    primary_result = float(operand_list[0]) * float(operand_list[1])
                elif operation == "-":
                    primary_result = float(operand_list[0]) - float(operand_list[1])
                elif operation == "+":
                    primary_result = float(operand_list[0]) + float(operand_list[1])
                for_str = for_str.replace(search_res, '%f' % (primary_result), 1)
            return for_str
        except Exception as e:
            logger.info(str(e))
            return None
    try:
        formula = primary_operator(formula, "/")
        formula = primary_operator(formula, "*")
        formula = primary_operator(formula, "-")
        formula = primary_operator(formula, "+")
    except Exception as e:
        logger.info(str(e))
        return None
    return formula


def remove_parentheses(formula):
    """
        去掉算式的括号，计算括号里算式
    :param formula: 
    :return: 
    """
    parentheses = re.compile("\([^()]+\)")
    while parentheses.search(formula):
        search_res = parentheses.search(formula).group()
        for_str = re.sub("[()]", "", search_res)
        if judge_illegal(for_str):
            return ""
        for_str = calculator_formula(for_str)
        formula = formula.replace(search_res, for_str, 1)
    """
    会有去掉所有括号算式还没算完的情况
    eg：1-2*65
    需要再计算一遍算式
    """
    formula = calculator_formula(formula)
    return formula


def result_formula(formula):
    """  
        简单计算器, 纯粹四则运算
        去完括号后额外计算的那一次若再次出现括号，
        则重复去括号运算，直至再没有括号
    :param formula: str
    :return: str
    """

    def remove_space(formula):
        """
            去掉算式的空格
        :param formula: str
        :return: str
        """
        return formula.replace(" ", "")

    def first_calculator(for_str):
        """
            先计算括号里边的
        :param for_str: 
        :return: 
        """
        if judge_illegal(for_str):
            return None
        return remove_parentheses(for_str)

    formula = remove_space(formula)

    formula = first_calculator(formula)
    if not formula:
        return None
    while has_parentheses(formula):
        formula = first_calculator(formula)
        # print("calculator_result:", formula)
    if not formula:
        return None
    return formula


if __name__ == '__main__':
    cal = result_formula("1+1+2+3*(35+1-5*7-10/5)/2*2")
    print(cal)
