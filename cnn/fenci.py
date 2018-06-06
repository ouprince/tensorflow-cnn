#-*- coding:utf-8 -*-
import jieba
import jieba.posseg as pseg
import time
import os
try:
    os.remove("yuliao_fenci.data")
except:pass

t1=time.time()
with open("chuli.data") as file_object:
    for line in file_object:
        seg_list=jieba.cut(line,cut_all=False)
        result=" ".join(seg_list)
        with open("yuliao_fenci.data","a") as writeme:
            writeme.write(result.encode("utf-8")+"\n")

t2=time.time()
print ("fen ci over :spend "+str(t2-t1)+" seconds")
