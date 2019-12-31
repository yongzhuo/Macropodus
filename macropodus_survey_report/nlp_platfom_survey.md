# 中文自然语言处理(nlp)工具调研与汇总(截至2019.11.16)


## 1.常见平台与功能
平台|语言|star|year|中文分词|词性标注|依存句法|实体识别|关键词提取|文本摘要|文本聚类|情感识别|文本相似|关系抽取|free|
---|---|---|---|---|---|---|---|---|---|---|---|---|---|---
jieba|python|20.8k|7/0.5|是|是|否|否|是|否|否|是|否|否|MIT
THULAC-Python|python|1.2k|4/1|是|是|否|否|否|否|否|否|否|否|MIT
pkuseg-python|python|4.3k|0.9/0.5|是|是|否|否|否|否|否|否|否|否|MIT
snownlp|python|4.4k|6/3/*|是|是|否|否|是|是|否|是|是|否|MIT
deepnlp|python|1.3k|2/2/!|是|是|是|是|是|是|否|否|否|否|MIT
fastNLP|python|0.9k|2/0|是|是|否|是|否|否|否|是|否|否|MIT
Jiagu|python|0.97k|0.9/0|是|是|是|是|是|是|是|是|否|是|MIT
YaYaNLP|python|0.05k|4/4/!|是|是|否|是|否|否|否|否|否|否|MIT
HanLP|java|16.4k|0.9/0|是|是|是|是|是|是|是|是|否|否|MIT
ansj-seg|java|5.2k|3/0.4|是|是|是|是|是|是|否|是|否|否|Apache-2.0
word|java|1.4k|5/1|是|是|否|是|否|否|否|否|是|否|Apache-2.0
Jcseg|java|0.69k|3/0|是|是|是|是|是|是|否|否|否|否|Apache-2.0
ik-analyzer|java|0.53k|9/9/!|是|是|是|否|否|否|否|否|否|否|LGPL-3.0
CoreNLP|java|6.7k|9/9/!|是|是|是|是|是|否|否|否|否|否|GUN2.0
fnlp|java|2.2k|6/0.9/!|是|是|是|是|是|是|是|否|否|否|LGPL-3.0
NLPIR|java|2.5k|?/1/!|是|是|否|否|否|否|是|否|否|否|not open
sego|go|1.2k|6/1/!|是|是|否|否|否|否|是|否|否|否|Apache-2.0
ltp|c++|2.3k|6/1/!|是|是|是|是|是|是|是|否|否|否|LGPL-3.0
PaddleNLP|c++|3.4k|6/1/!|是|是|是|是|是|是|是|是|是|是|Apache-2.0


##备注
* 1.year中"6/3/*"表示"项目开始时间/最近更新时间/在维护";!表示不维护,超过一年不维护,不回复issiue则认为放弃;
* 2.其他功能
    * snownlp: 拼音转换,繁简转换,tf-idf计算,切句子
    * deepnlp: tensorflow1.4训练的各种模型
    * NLPIR: 检索,敏感信息,文档去重,编码转换
    * Ltp: 事件抽取,srl,时间抽取,
    * HanLP: 人民日报2014分词,文本推荐(相似度),索引分词
    * ansj-seg: 比较混乱,主页没有调用说明,词典是个大杂烩
    * word: 词频统计、词性标注、同义标注、反义标注、拼音标注
    * ltp: 特征裁剪策略,语义角色标注
    * PaddleNLP: Paddle训练,以及基础包,enienr生成等各种任务
* 3.更多的统计学习方法
    摘要,情感识别(酸甜苦辣),新词发现,实体与关系抽取,领域分类,生成
    
    
##分词算法
* 1.jieba
   * 1.1 基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图 (DAG)
   * 1.2 采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合
   * 1.3 对于未登录词，采用了基于汉字成词能力的 HMM 模型，使用了 Viterbi 算法
* 2.THULAC,pkuseg,Jiagu,fastNLP
   * 2.1 CRF(char,word,elmo,bert)
   * 2.2 feature+CRF
* 3.ansj-seg
   * 3.1 n-Gram+CRF+HMM
* 4.HanLP
   * 4.1 n-Gram, CRF
* 5.sego
   * 5.1 基于词频的最短路径加动态规划
* 6.Ltp
   * 6.1 bilstm+crf
   * 6.2    英文、URI一类特殊词识别规则
            利用空格等自然标注线索
            在统计模型中融入词典信息
            从大规模未标注数据中统计的字间互信息、上下文丰富程度
* 7.PaddleNLP
   * 7.1 gru+crf
* 8.word(最大匹配法、最大概率法、最短路径法)
   * 8.1 正向最大匹配算法,逆向最大匹配算法,正向最小匹配算法,逆向最小匹配算法
   * 8.2 双向最大匹配算法,双向最小匹配算法,双向最大最小匹配算法
   * 8.3 全切分算法,最少词数算法,最大Ngram分值算法,最短路径法
   * 8.4 语义切分:扩充转移网络法、知识分词语义分析法、邻接约束法、综合匹配法、后缀分词法、特征词库法、矩阵约束法、语法分析法


## 工具包地址
* jiba:[https://github.com/fxsjy/jieba](https://github.com/fxsjy/jieba)
* HanLP:[https://github.com/hankcs/HanLP](https://github.com/hankcs/HanLP)
* CoreNLP:[https://github.com/stanfordnlp/CoreNLP](https://github.com/stanfordnlp/CoreNLP)
* ansj-seg:[https://github.com/lionsoul2014/jcseg](https://github.com/lionsoul2014/jcseg)
* THULAC-Python:[https://github.com/thunlp/THULAC-Python](https://github.com/thunlp/THULAC-Python)
* pkuseg-python:[https://github.com/lancopku/pkuseg-python](https://github.com/lancopku/pkuseg-python)
* snownlp:[https://github.com/isnowfy/snownlp](https://github.com/isnowfy/snownlp)
* deepnlp:[https://github.com/rockingdingo/deepnlp](https://github.com/rockingdingo/deepnlp)
* fastNLP:[https://github.com/fastnlp/fastNLP](https://github.com/fastnlp/fastNLP)
* Jiagu:[https://github.com/ownthink/Jiagu](https://github.com/ownthink/Jiagu)
* xmnlp:[https://github.com/SeanLee97/xmnlp](https://github.com/SeanLee97/xmnlp)
* word:[https://github.com/ysc/word](https://github.com/ysc/word)
* jcseg:[https://github.com/lionsoul2014/jcseg](https://github.com/lionsoul2014/jcseg)
* paddleNLP:[https://github.com/PaddlePaddle/models](https://github.com/PaddlePaddle/models)
* sego:[https://github.com/huichen/sego](https://github.com/huichen/sego)
* ik-analyzer:[https://github.com/wks/ik-analyzer](https://github.com/wks/ik-analyzer)
* fnlp:[https://github.com/FudanNLP/fnlp](https://github.com/FudanNLP/fnlp)
* NLPIR:[https://github.com/NLPIR-team/NLPIR](https://github.com/NLPIR-team/NLPIR)
