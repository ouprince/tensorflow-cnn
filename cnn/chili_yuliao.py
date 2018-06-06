#-*- coding:utf-8 -*-
import re,sys,os
import time
t1 = time.time()
read_file = "train.raw.data"
write_file = "y_train"

reload(sys)
sys.setdefaultencoding("utf-8")

try:
    os.remove(write_file)
except:
    pass

def write_to_file(line):
    with open(write_file,'a') as write_me:
        write_me.write(line.encode("utf-8") + "\n")

def del_punct(line):
    line = line.decode("utf8")
    string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[.+——！，。？?、~@#￥%……&* “” 《》【】（）：() ；01]+".decode("utf8"), "".decode("utf8"),line)
    return string

if __name__ == "__main__":
    i = 0
    with open(read_file) as read_me:
        for lines in read_me:
            write_to_file(del_punct(lines))
            i += 1
            if i % 1000 == 0:
                t2 = time.time()
                print ("%d rows writed in %s,spend %d seconds" %(i,write_file,int(t2-t1)))
