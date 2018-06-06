#-*- coding:utf-8 -*-
import gensim
import logging
import os
import time
from gensim.models import word2vec

t1=time.time()
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
sentences = []
with open('yuliao_fenci.data','r') as readme:
    for line in readme.readlines():
        sentences.append(line.split())

t2=time.time()
print ('读取文件到内存并转换耗时:'+str(t2-t1)+"s")

#model = gensim.models.Word2Vec(sentences, min_count=1)
model = word2vec.Word2Vec(sentences, size=60, window=5, min_count=2)
model.save('model_oyp')
