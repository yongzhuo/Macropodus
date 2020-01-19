# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/11/19 20:40
# @author   :Mo
# @function :TrieTree of keywords find, 只返回查全的情况, 查找句子中的关键词（例如影视名、人名、关键词、实体等）


from macropodus.conf.path_log import get_logger_root


logger = get_logger_root()


class TrieNode:
    """
        前缀树节点-链表
    """
    def __init__(self):
        self.child = {}


class TrieTree:
    """
        前缀树构建, 新增关键词, 关键词词语查找等
    """
    def __init__(self):
        self.algorithm = "trietree"
        self.root = TrieNode()

    def add_keyword(self, keyword):
        """
            新增一个关键词
        :param keyword: str, 构建的关键词
        :return: None
        """
        node_curr = self.root
        for word in keyword:
            if node_curr.child.get(word) is None:
                node_next = TrieNode()
                node_curr.child[word] = node_next
            node_curr = node_curr.child[word]
        # 每个关键词词后边, 加入end标志位
        if node_curr.child.get('[END]') is None:
            node_next = TrieNode()
            node_curr.child['[END]'] = node_next
        node_curr = node_curr.child['[END]']
        logger.info("add {} success!".format("".join(keyword)))

    def delete_keyword(self, keyword):
        """
            删除一个关键词
        :param keyword: str, 构建的关键词
        :return: None
        """
        node_curr = self.root
        flag = 1
        for word in keyword:
            if node_curr.child.get(word) is not None:
                node_curr = node_curr.child[word]
            else:
                flag = 0
        # 每个关键词词后边, 加入end标志位
        if node_curr.child.get('[END]') is not None and flag == 1:
            node_curr.child.pop('[END]')
        else:
            logger.info("{} is not in trietree, delete keyword faild!".format("".join(keyword)))

    def add_keywords_from_list(self, keywords):
        """
            新增关键词s, 格式为list
        :param keyword: list, 构建的关键词
        :return: None
        """
        for keyword in keywords:
            self.add_keyword(keyword)

    def find_keyword(self, sentence):
        """
            从句子中提取关键词, 可提取多个
        :param sentence: str, 输入的句子
        :return: list, 提取到的关键词
        """
        assert type(sentence) == str
        if not sentence: # 空格字符不取
            return []

        node_curr = self.root # 关键词的头, 每遍历完一遍后需要重新初始化
        index_last = len(sentence)
        keyword_list = []
        keyword = ''
        count = 0
        for word in sentence:
            count += 1
            if node_curr.child.get(word) is None: # 查看有无后缀, 即匹配到一个关键词最后一个字符的时候
                if keyword: # 提取到的关键词(也可能是前面的几位)
                    if node_curr.child.get('[END]') is not None: # 取以end结尾的关键词
                        keyword_list.append(keyword)
                    if self.root.child.get(word) is not None: # 处理连续的关键词情况, 如"第九区流浪地球"
                        keyword = word
                        node_curr = self.root.child[word]
                    else: #
                        keyword = ''
                        node_curr = self.root  # 重新初始化
            else: # 有后缀就加到name里边
                keyword = keyword + word
                node_curr = node_curr.child[word]
                if count == index_last:  # 实体结尾的情况
                    if node_curr.child.get('[END]') is not None:
                        keyword_list.append(keyword)
        return keyword_list

    def match_keyword(self, keyword):
        """
            判断keyword在不在trietree里边
        :param keyword: str, input word
        :return: boolean, True or False
        """
        node = self.root
        for kw in keyword:
            if not node.child.get(kw):
                return False
            node = node.child[kw]
        if not node.child.get('[END]'):
            return False
        return True


def get_trie_tree_class(keywords):
    """
        根据list关键词，初始化trie树
    :param keywords: list, input
    :return: objext, 返回实例化的trie
    """
    trie = TrieTree()
    trie.add_keywords_from_list(keywords)
    return trie


if __name__ == "__main__":
    print("".join("你好呀"))
    # 测试1, class实例
    trie = TrieTree()
    keywords = ['英雄', '人在囧途', '那些年,我们一起追过的女孩', '流浪地球', '华娱',
                '犬夜叉', '火影', '名侦探柯南', '约会大作战', '名作之壁', '动漫',
                '乃木坂46', 'akb48', '飘', '最后的武士', '约会', '英雄2', '日娱',
                '2012', '第九区', '星球大战', '侏罗纪公园', '泰坦尼克号', 'Speed']
    keywords = [list(keyword.strip()) for keyword in keywords]
    trie.add_keywords_from_list(keywords) # 创建树
    keyword = trie.find_keyword('第九区约会, 侏罗纪公园和泰坦尼克号泰坦尼克号')
    print(keyword)
    gg = trie.delete_keyword('英雄')
    gg = trie.delete_keyword('英雄3')

    keyword = trie.match_keyword('英雄')
    keyword2 = trie.match_keyword('英雄2')

    print(keyword)


    # 测试2, get树
    trie_tree = get_trie_tree_class(keywords) # 创建树并返回实例化class
    while True:
        print("sihui请你输入:")
        input_ques = input()
        keywords = trie_tree.find_keyword(input_ques)
        print(keywords)
