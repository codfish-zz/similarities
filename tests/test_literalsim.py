# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import sys
import unittest

sys.path.append('..')

from similarities.literalsim import SimhashSimilarity, TfidfSimilarity, BM25Similarity, WordEmbeddingSimilarity, \
    CilinSimilarity, HownetSimilarity
from text2vec import Word2Vec


class LiteralCase(unittest.TestCase):
    def test_simhash(self):
        """test_simhash"""
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        m = SimhashSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        print(m.most_similar('刘若英是演员'))
        self.assertEqual(len(m.most_similar('刘若英是演员')), 0)
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        m.add_corpus(zh_list)
        r = m.most_similar('刘若英是演员', topn=2)
        print(r)
        self.assertAlmostEqual(m.similarity(text1, text2), 0.734375, places=4)
        self.assertEqual(len(r), 2)

    def test_tfidf(self):
        """test_tfidf"""
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        m = TfidfSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '我不是演员吗']
        m.add_corpus(zh_list)
        print(m.most_similar('刘若英是演员'))
        self.assertEqual(len(m.most_similar('刘若英是演员')), 4)

    def test_bm25(self):
        """test_bm25"""
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        m = BM25Similarity()
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '我不是演员吗']
        m.add_corpus(zh_list)
        print(m.most_similar('刘若英是演员'))
        self.assertEqual(len(m.most_similar('刘若英是演员')), 4)

    def test_word2vec(self):
        """test_word2vec"""
        text1 = '刘若英是个演员'
        text2 = '他唱歌很好听'
        wm = Word2Vec()
        list_of_corpus = ["This is a test1", "This is a test2", "This is a test3"]
        list_of_corpus2 = ["that is test4", "that is a test5", "that is a test6"]
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '刘若英是个演员', '演戏很好看的人']
        m = WordEmbeddingSimilarity(wm, list_of_corpus)
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        m.add_corpus(list_of_corpus2+zh_list)
        v = m.get_vector("This is a test1")
        print(v[:10], v.shape)
        print(m.similarity("This is a test1", "that is a test5"))
        print(m.distance("This is a test1", "that is a test5"))
        print(m.most_similar("This is a test1"))
        print(m.most_similar("刘若英是演员"))
        self.assertEqual(len(m.most_similar('刘若英是演员', topn=6)), 6)

    def test_cilin(self):
        """test_cilin"""
        text1 = '周杰伦是一个歌手'
        text2 = '刘若英是个演员'
        m = CilinSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        m.add_corpus(zh_list)
        print(m.most_similar('刘若英是演员'))
        self.assertEqual(len(m.most_similar('刘若英是演员')), 3)

    def test_hownet(self):
        """test_cilin"""
        text1 = '周杰伦是一个歌手'
        text2 = '刘若英是个演员'
        m = HownetSimilarity()
        print(m.similarity(text1, text2))
        print(m.distance(text1, text2))
        zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
        m.add_corpus(zh_list)
        print(m.most_similar('刘若英是演员'))
        self.assertEqual(len(m.most_similar('刘若英是演员')), 3)


if __name__ == '__main__':
    unittest.main()