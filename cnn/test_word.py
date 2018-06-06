# -*- coding:utf-8 -*-
import gensim
import logging,sys
from gensim.models import Word2Vec
model=gensim.models.Word2Vec.load("word_model2/Word60.model")
logging.basicConfig(format='%(asctime)s: %(levelname)s :%(message)s',filename='my_log.log',level=logging.INFO)

reload(sys)
sys.setdefaultencoding("utf-8")
for word in model.index2word:
    print word
'''
ss = '瘟疫'

sim = model.most_similar(unicode(ss))
for i in sim:
    print i[0]
'''
