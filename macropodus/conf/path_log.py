# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/11/18 23:59
# @author  : Mo
# @function: logger of macropodus


from macropodus.conf.path_config import path_log_basic
from logging.handlers import RotatingFileHandler
import logging
import time
import os


logger_level = logging.INFO
# log目录地址
path_logs = path_log_basic #  + '/logs'
if not os.path.exists(path_logs):
    os.mkdir(path_logs)
# 全局日志格式
logging.basicConfig(level=logger_level,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
# 定义一个日志记录器
logger = logging.getLogger("macropodus")
logger.setLevel(level = logger_level)
# 日志文件名,为启动时的日期
log_file_name = time.strftime('macropodus-%Y-%m-%d', time.localtime(time.time())) + ".log"
log_name_day = os.path.join(path_logs, log_file_name)
# 文件输出, 定义一个RotatingFileHandler，最多备份32个日志文件，每个日志文件最大32K
fHandler = RotatingFileHandler(log_name_day, maxBytes = 32*1024, backupCount = 32)
fHandler.setLevel(logger_level)
# 日志输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fHandler.setFormatter(formatter)
# # 控制台输出
# console = logging.StreamHandler()
# console.setLevel(logger_level)
# console.setFormatter(formatter)
# logger加到handel里边
logger.addHandler(fHandler)
# logger.addHandler(console)


def get_logger_root(name="macropodus"):
    """
        全局日志引用
    :param name: str, name of logger
    :return: object, logging
    """
    return logging.getLogger(name)
