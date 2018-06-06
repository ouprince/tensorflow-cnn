#-*- coding:utf-8 -*-
import sys
import gensim
import logging
import numpy as np
import jieba
import jieba.posseg as pseg
from chili_yuliao import del_punct

reload(sys)
sys.setdefaultencoding("utf-8")
model=gensim.models.Word2Vec.load("word_model2/Word60.model")
logging.basicConfig(format='%(asctime)s: %(levelname)s :%(message)s',filename='my_log.log',level=logging.INFO)

zeros = []
for i in range(60):
    zeros.append(0)
def trands(x):
    if x == -1:
        return [1,0,0]
    elif x == 0:
        return [0,1,0]
    elif x == 1:
        return [0,0,1]
    else:
        raise KeyError

def data_load(data_file):
    xs = []
    ys = []
    with open(data_file) as readme:
        for lines in readme:
            xs.append(trands(int(lines[0:2].replace("/t",""))))
            xinxi = " ".join(jieba.cut(del_punct(lines[2:]),cut_all=False)).split(" ")[0:16*16]
            while len(xinxi) < 16*16:
                xinxi.append("nothing")
            result = []
            for s in xinxi:
                try:
                    result.append(model[unicode(s)])
                except: result.append(zeros)
            ys.append(np.reshape(result,[16,16,60]))
    return xs,ys
