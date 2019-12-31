# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/8/21 22:01
# @author   :Mo
# @function :textrank using gensim summarization of chinese. (split is '. ', '! ', '. ' and so on)
# @code from:most code from https://github.com/RaRe-Technologies/gensim

# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
# """This module provides functions for summarizing texts. Summarizing is based on
# ranks of text sentences using a variation of the TextRank algorithm [1]_.
#
# .. [1] Federico Barrios, Federico L´opez, Luis Argerich, Rosita Wachenchauzer (2016).
#        Variations of the Similarity Function of TextRank for Automated Summarization,
#        https://arxiv.org/abs/1602.03606


from gensim.summarization.commons import remove_unreachable_nodes as _remove_unreachable_nodes
from gensim.summarization.pagerank_weighted import pagerank_weighted as _pagerank
from gensim.summarization.commons import build_graph as _build_graph
from gensim.summarization.bm25 import iter_bm25_bow as _bm25_weights
from gensim.corpora import Dictionary
from gensim.utils import deprecated
from math import log10 as _log10
from six.moves import range
import logging

from macropodus.data.words_common.stop_words import stop_words
from macropodus.preprocess.tools_ml import macropodus_cut
from macropodus.preprocess.tools_ml import cut_sentence


logger = logging.getLogger(__name__)
WEIGHT_THRESHOLD = 1.e-3
INPUT_MIN_LENGTH = 2


class TextrankGensimSum:
      def __init__(self):
        self.algorithm = 'textrank_gensim'
        self.stop_words = stop_words.values()
        self.len_ideal = 18 # 中心句子长度, 默认

      def summarize(self, text, num=320):
            # 切句
            if type(text) == str:
                sentences = cut_sentence(text)
            elif type(text) == list:
                sentences = text
            else:
                raise RuntimeError("text type must be list or str")
            # str of sentence >>> index
            corpus = _build_corpus(sentences)
            # pagerank and so on
            most_important_docs = summarize_corpus(corpus)

            count = 0
            sentences_score = {}
            for cor in corpus:
                tuple_cor = tuple(cor)
                sentences_score[sentences[count]] = most_important_docs[tuple_cor]
                count += 1
            # 最小句子数
            num_min = min(num, int(len(sentences) * 0.6))
            score_sen = [(rc[1], rc[0]) for rc in sorted(sentences_score.items(),
                                                         key=lambda d: d[1], reverse=True)][0:num_min]
            return score_sen



def _set_graph_edge_weights(graph):
    """Sets weights using BM25 algorithm. Leaves small weights as zeroes. If all weights are fairly small,
     forces all weights to 1, inplace.

    Parameters
    ----------
    graph : :class:`~gensim.summarization.graph.Graph`
        Given graph.

    """
    documents = graph.nodes()
    weights = _bm25_weights(documents)

    for i, doc_bow in enumerate(weights):
        if i % 1000 == 0 and i > 0:
            logger.info('PROGRESS: processing %s/%s doc (%s non zero elements)', i, len(documents), len(doc_bow))

        for j, weight in doc_bow:
            if i == j or weight < WEIGHT_THRESHOLD:
                continue

            edge = (documents[i], documents[j])

            if not graph.has_edge(edge):
                graph.add_edge(edge, weight)

    # Handles the case in which all similarities are zero.
    # The resultant summary will consist of random sentences.
    if all(graph.edge_weight(edge) == 0 for edge in graph.iter_edges()):
        _create_valid_graph(graph)


def _create_valid_graph(graph):
    """Sets all weights of edges for different edges as 1, inplace.

    Parameters
    ----------
    graph : :class:`~gensim.summarization.graph.Graph`
        Given graph.

    """
    nodes = graph.nodes()

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j:
                continue

            edge = (nodes[i], nodes[j])

            if graph.has_edge(edge):
                graph.del_edge(edge)

            graph.add_edge(edge, 1)


@deprecated("Function will be removed in 4.0.0")
def _get_doc_length(doc):
    """Get length of (tokenized) document.

    Parameters
    ----------
    doc : list of (list of (tuple of int))
        Given document.

    Returns
    -------
    int
        Length of document.

    """
    return sum(item[1] for item in doc)


@deprecated("Function will be removed in 4.0.0")
def _get_similarity(doc1, doc2, vec1, vec2):
    """Returns similarity of two documents.

    Parameters
    ----------
    doc1 : list of (list of (tuple of int))
        First document.
    doc2 : list of (list of (tuple of int))
        Second document.
    vec1 : array
        ? of first document.
    vec1 : array
        ? of secont document.

    Returns
    -------
    float
        Similarity of two documents.

    """
    numerator = vec1.dot(vec2.transpose()).toarray()[0][0]
    length_1 = _get_doc_length(doc1)
    length_2 = _get_doc_length(doc2)

    denominator = _log10(length_1) + _log10(length_2) if length_1 > 0 and length_2 > 0 else 0

    return numerator / denominator if denominator != 0 else 0


def _build_corpus(sentences):
    """Construct corpus from provided sentences.

    Parameters
    ----------
    sentences : list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Given sentences.

    Returns
    -------
    list of list of (int, int)
        Corpus built from sentences.

    """
    split_tokens = [macropodus_cut(sentence) for sentence in sentences]
    dictionary = Dictionary(split_tokens)
    return [dictionary.doc2bow(token) for token in split_tokens]


def _get_important_sentences(sentences, corpus, important_docs):
    """Get most important sentences.

    Parameters
    ----------
    sentences : list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Given sentences.
    corpus : list of list of (int, int)
        Provided corpus.
    important_docs : list of list of (int, int)
        Most important documents of the corpus.

    Returns
    -------
    list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Most important sentences.

    """
    hashable_corpus = _build_hasheable_corpus(corpus)
    sentences_by_corpus = dict(zip(hashable_corpus, sentences))
    return [sentences_by_corpus[tuple(important_doc)] for important_doc in important_docs]


def _get_sentences_with_word_count(sentences, word_count):
    """Get list of sentences. Total number of returned words close to specified `word_count`.

    Parameters
    ----------
    sentences : list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Given sentences.
    word_count : int or None
        Number of returned words. If None full most important sentences will be returned.

    Returns
    -------
    list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Most important sentences.

    """
    length = 0
    selected_sentences = []

    # Loops until the word count is reached.
    for sentence in sentences:
        words_in_sentence = len(sentence.text.split())

        # Checks if the inclusion of the sentence gives a better approximation
        # to the word parameter.
        if abs(word_count - length - words_in_sentence) > abs(word_count - length):
            return selected_sentences

        selected_sentences.append(sentence)
        length += words_in_sentence

    return selected_sentences


def _extract_important_sentences(sentences, corpus, important_docs, word_count):
    """Get most important sentences of the `corpus`.

    Parameters
    ----------
    sentences : list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Given sentences.
    corpus : list of list of (int, int)
        Provided corpus.
    important_docs : list of list of (int, int)
        Most important docs of the corpus.
    word_count : int
        Number of returned words. If None full most important sentences will be returned.

    Returns
    -------
    list of :class:`~gensim.summarization.syntactic_unit.SyntacticUnit`
        Most important sentences.

    """
    important_sentences = _get_important_sentences(sentences, corpus, important_docs)

    # If no "word_count" option is provided, the number of sentences is
    # reduced by the provided ratio. Else, the ratio is ignored.
    return important_sentences \
        if word_count is None \
        else _get_sentences_with_word_count(important_sentences, word_count)


def _format_results(extracted_sentences, split):
    """Returns `extracted_sentences` in desired format.

    Parameters
    ----------
    extracted_sentences : list of :class:~gensim.summarization.syntactic_unit.SyntacticUnit
        Given sentences.
    split : bool
        If True sentences will be returned as list. Otherwise sentences will be merged and returned as string.

    Returns
    -------
    list of str
        If `split` **OR**
    str
        Formatted result.

    """
    if split:
        return [sentence for sentence in extracted_sentences]
    return "\n".join(sentence.text for sentence in extracted_sentences)


def _build_hasheable_corpus(corpus):
    """Hashes and get `corpus`.

    Parameters
    ----------
    corpus : list of list of (int, int)
        Given corpus.

    Returns
    -------
    list of list of (int, int)
        Hashable corpus.

    """
    return [tuple(doc) for doc in corpus]


def summarize_corpus(corpus):
    """Get a list of the most important documents of a corpus using a variation of the TextRank algorithm [1]_.
     Used as helper for summarize :func:`~gensim.summarization.summarizer.summarizer`

    Note
    ----
    The input must have at least :const:`~gensim.summarization.summarizer.INPUT_MIN_LENGTH` documents for the summary
    to make sense.


    Parameters
    ----------
    corpus : list of list of (int, int)
        Given corpus.
    ratio : float, optional
        Number between 0 and 1 that determines the proportion of the number of
        sentences of the original text to be chosen for the summary, optional.

    Returns
    -------
    list of str
        Most important documents of given `corpus` sorted by the document score, highest first.

    """
    hashable_corpus = _build_hasheable_corpus(corpus)

    logger.info('Building graph')
    graph = _build_graph(hashable_corpus)

    logger.info('Filling graph')
    _set_graph_edge_weights(graph)

    logger.info('Removing unreachable nodes of graph')
    _remove_unreachable_nodes(graph)

    logger.info('Pagerank graph')
    pagerank_scores = _pagerank(graph)
    return pagerank_scores

    # logger.info('Sorting pagerank scores')
    # hashable_corpus.sort(key=lambda doc: pagerank_scores.get(doc, 0), reverse=True)
    #
    # return [list(doc) for doc in hashable_corpus]

