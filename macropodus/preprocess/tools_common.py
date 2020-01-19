# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/19 0:15
# @author  : Mo
# @function: common tools of macropodus


from macropodus.conf.path_log import get_logger_root
import json
import os
import re


re_continue = re.compile("[A-Za-z0-9.@_]", re.U)
re_zh_cn = re.compile("([\u4E00-\u9FD5]+)", re.U)


logger = get_logger_root()


__all__ = ["txt_read",
           "txt_write",
           "save_json",
           "load_json",
           "delete_file"]


def txt_read(path_file, encode_type='utf-8'):
    """
        读取txt文件，默认utf8格式, 不能有空行
    :param file_path: str, 文件路径
    :param encode_type: str, 编码格式
    :return: list
    """
    list_line = []
    try:
        file = open(path_file, 'r', encoding=encode_type)
        while True:
            line = file.readline().strip()
            if not line:
                break
            list_line.append(line)
        file.close()
    except Exception as e:
        logger.info(str(e))
    finally:
        return list_line


def txt_write(list_line, file_path, type='w', encode_type='utf-8'):
    """
      txt写入list文件
    :param listLine:list, list文件，写入要带"\n" 
    :param filePath:str, 写入文件的路径
    :param type: str, 写入类型, w, a等
    :param encode_type: 
    :return: 
    """
    try:
        file = open(file_path, type, encoding=encode_type)
        file.writelines(list_line)
        file.close()
    except Exception as e:
        logger.info(str(e))


def save_json(json_lines, json_path):
    """
      保存json，
    :param json_lines: json 
    :param path: str
    :return: None
    """
    with open(json_path, 'w', encoding='utf-8') as fj:
        fj.write(json.dumps(json_lines, ensure_ascii=False))
    fj.close()


def load_json(path):
    """
      获取json, json存储为[{}]格式, like [{'大漠帝国':132}]
    :param path: str
    :return: json
    """
    with open(path, 'r', encoding='utf-8') as fj:
        model_json = json.load(fj)
    return model_json


def delete_file(path):
    """
        删除一个目录下的所有文件
    :param path: str, dir path
    :return: None
    """
    for i in os.listdir(path):
        # 取文件或者目录的绝对路径
        path_children = os.path.join(path, i)
        if os.path.isfile(path_children):
            if path_children.endswith(".h5") or path_children.endswith(".json") or "events" in path_children or "trace" in path_children:
                os.remove(path_children)
        else:# 递归, 删除目录下的所有文件
            delete_file(path_children)


def get_dir_files(path_dir):
    """
        递归获取某个目录下的所有文件(单层)
    :param path_dir: str, like '/home/data'
    :return: list, like ['2019_12_5.txt']
    """

    def get_dir_files_func(file_list, dir_list, root_path=path_dir):
        """
            递归获取某个目录下的所有文件
        :param root_path: str, like '/home/data'
        :param file_list: list, like []
        :param dir_list: list, like []
        :return: None
        """
        # 获取该目录下所有的文件名称和目录名称
        dir_or_files = os.listdir(root_path)
        for dir_file in dir_or_files:
            # 获取目录或者文件的路径
            dir_file_path = os.path.join(root_path, dir_file)
            # 判断该路径为文件还是路径
            if os.path.isdir(dir_file_path):
                dir_list.append(dir_file_path)
                # 递归获取所有文件和目录的路径
                get_dir_files_func(dir_file_path, file_list, dir_list)
            else:
                file_list.append(dir_file_path)

    # 用来存放所有的文件路径
    _files = []
    # 用来存放所有的目录路径
    dir_list = []
    get_dir_files_func(_files, dir_list, path_dir)
    return _files


def get_all_dirs_files(path_dir):
    """
        递归获取某个目录下的所有文件(所有层, 包括子目录)
    :param path_dir: str, like '/home/data'
    :return: list, like ['2020_01_08.txt']
    """
    path_files = []
    def get_path_files(path_dir):
        """
            递归函数, 获取某个目录下的所有文件
        :param path_dir: str, like '/home/data'
        :return: list, like ['2020_01_08.txt']
        """
        for root, dirs, files in os.walk(path_dir):
            for fi in files: # 递归的终止条件
                path_file = os.path.join(root, fi)
                path_files.append(path_file)
            for di in dirs:  # 语间目录便继续递归
                path_dir = os.path.join(root, di)
                get_path_files(path_dir)
    get_path_files(path_dir)
    return path_files
