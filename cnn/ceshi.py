#-*- coding:utf-8 -*-
import codecs
from textrank4zh import TextRank4Keyword, TextRank4Sentence
#texts = codecs.open('test.data', 'r', 'utf-8').read()
tr4s = TextRank4Sentence(stop_words_file='stopwords.txt')  # 导入停止词
tr4w = TextRank4Keyword(stop_words_file = 'stopwords.txt')

with open("test.data") as readme:
    for ttt in readme:
        tr4s.analyze(text=ttt, lower=True, source = 'all_filters')
        tr4w.analyze(text=ttt,lower=True, window=4, pagerank_config={'alpha':0.85})
        key = ""
        print "关键词:"
        for item in tr4w.get_keywords(num=10, word_min_len=2):
            key += item.word + " "
        print key

        count_sentens = 0
        print "摘要:"
        for item in tr4s.get_key_sentences(num=50):
            if float(item.weight) >= 0.046:
                count_sentens += 1
                print(item.sentence) #type(item.sentence)
            else:
                break

        if count_sentens == 0:
            for item in tr4s.get_key_sentences(num=6):
                print(item.sentence) 
