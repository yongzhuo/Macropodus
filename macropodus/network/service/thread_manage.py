# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2020/1/16 15:08
# @author  : Mo
# @function:


from multiprocessing import Process, Manager

def f(d, l):
    d[1] = '1'
    d['2'] = 2
    d[0.25] = None
    l.reverse()

if __name__ == '__main__':
    manager = Manager()

    d = manager.dict()
    l = manager.list(range(10))

    p = Process(target=f, args=(d, l))
    p.start()
    p.join()

    print (d)
    print (l)