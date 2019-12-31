# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2019/12/17 21:13
# @author  : Mo
# @function: test evulate


from macropodus.preprocess.tools_common import txt_write, txt_read
import macropodus
import time


def evulate_file(path_file):
	"""
	    验证切词的各种指标
	:param path_file: str, like '/train.txt'
	:return: float
	"""
	# 读取数据
	sents = txt_read(path_file)
	# 初始化统计计数
	count_macropodus = 0
	count_real = 0
	count_true = 0
	count = 0
	# 切词与统计, true
	for sent in sents:
		sent_sp = sent.strip()
		res_real = sent_sp.split(' ')
		sentence = sent_sp.replace(' ','')
		res_macropodus = macropodus.cut(sentence)
		print(res_macropodus)
		count += 1
		count_real += len(res_real)
		count_macropodus += len(res_macropodus)
		for cm in res_macropodus:
			if cm in res_real:
				count_true += 1
				res_real.remove(cm)
	# precision, recall, f1
	precision = count_true / count_macropodus
	recall = count_true / count_real
	f1 = (precision * recall * 2) / (precision + recall)

	return precision, recall, f1


if __name__ == "__main__":
	path_file = 'data/ambiguity.txt'
	time_start = time.time()
	precision, recall, f1 = evulate_file(path_file)
	print('time: ' + str(time.time()-time_start))
	print('precision\t', 'recall\t', 'f1')
	print(precision, recall, f1)
