# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/12/30 22:17
# @author   :Mo
# @function :setup of Macropodus
# @codes    :fix it and copy reference from https://github.com/TianWenQAQ/Kashgari/blob/master/setup.py


from macropodus.version import __version__
from setuptools import find_packages, setup
import codecs


# Package meta-data.
NAME = 'Macropodus'
DESCRIPTION = 'Macropodus: Tookit of Chinese Natural Language Processing'
URL = 'https://github.com/yongzhuo/Macropodus'
EMAIL = '1903865025@qq.com'
AUTHOR = 'yongzhuo'
LICENSE = 'MIT'

with codecs.open('README.md', 'r', 'utf8') as reader:
    long_description = "\n".join(reader.readlines())
with codecs.open('requirements.txt', 'r', 'utf8') as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))

setup(name=NAME,
        version=__version__,
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        packages=find_packages(exclude=('test')),
      package_data={'macropodus': ['*.*', 'data/*', 'data/dict/*',
                                   'data/embedding/*', 'data/embedding/word2vec/*',
                                   'data/model/*']
                    },
        install_requires=install_requires,
        license=LICENSE,
        classifiers=['License :: OSI Approved :: MIT License',
                     'Programming Language :: Python :: 3.5',
                     'Programming Language :: Python :: 3.6',
                     'Programming Language :: Python :: 3.7',
                     'Programming Language :: Python :: 3.8',
                     'Programming Language :: Python :: Implementation :: CPython',
                     'Programming Language :: Python :: Implementation :: PyPy'],
      )


if __name__ == "__main__":
    print("setup ok!")

# 说明, tensorflow>=1.13.0 or tensorflow-gpu>=1.13.0
# 项目工程目录这里Macropodus, 实际上, 下边还要有一层macropodus, 也就是说, macropodus和setup同一层
# data包里必须要有__init__.py, 否则文件不会生成, .py文件才能copy
# 编译的2种方案:

# 方案一
#     打开cmd
#     到达安装目录
#     python setup.py build
#     python setup.py install

# 方案二
# python setup.py bdist_wheel --universal
# twine upload dist/*

#
# conda remove -n py35 --all
# conda create -n py351 python=3.5

