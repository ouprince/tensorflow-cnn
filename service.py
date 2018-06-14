#-*- coding:utf-8 -*-
import sys,os
reload(sys)
sys.setdefaultencoding('utf8')
import json
from flask import Flask,request
from flask import Response
from flask_cors import CORS
import cPickle
curdir = os.path.dirname(os.path.abspath(__file__))
submit_data = os.path.join(curdir,"submit_data")
result_data = os.path.join(curdir,"result_data")
rootdir = os.path.join(curdir,os.path.pardir,os.path.pardir)
sys.path.append(rootdir)
from chinese_whispers.app.common.tokenizer_api import fenci_cut
from chinese_whispers.app.common.similarity import compare,sentence_vector
from page_rank.TextRank4ZH_master.textrank4zh.compute_yingda import get_word_similar,word_neighbors,word_model,word_cloud
from page_rank.TextRank4ZH_master.textrank import textrank
from text_rank.TextRank4ZH_master.textrank_yuan import textrank_yuan
from svm_article.predict import predict as predict_article
from svm_sentence.predict import predict as predict_comment

already_id = set()
already_get = set()
taskid_task = 106

app = Flask(__name__)
CORS(app, resources=r'/*')
@app.route('/cw',methods=['POST'])
def hello_world():
    global taskid_task
    global already_id
    taskid_task += 1
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)

    data = request.get_data()
    data = json.loads(data)
    data = data["data"]
    fw = open(os.path.join(submit_data,str(taskid_task)),"w")
    cPickle.dump(data,fw)
    fw.close()

    already_id.add(int(taskid_task))
    res = """{"task-id":%d}""" %taskid_task
    resp = Response(res)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
    #return ("""{"task-id":%d}""" %taskid_task)

@app.route('/cw/<int:task_id>',methods=['GET'])
def get_res(task_id):
    global already_get
    global taskid_task
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)

    for _,_,taskids in os.walk(result_data):
        for taskid in taskids:
            if int(taskid) == int(task_id):
                fw = codecs.open(os.path.join(result_data,taskid),"r","utf-8").read()
                #os.remove(os.path.join(result_data,taskid))
                already_get.add(int(task_id))
                if len(fw.strip()) == 0:
                    fw = json.dumps({"status":"no_data_enough","result":"no_result"})
                return fw

    res = ''
    if int(task_id) in already_get:
        res = json.dumps({"status":"already get","result":"failed"})
    elif int(task_id) > taskid_task:
        res = json.dumps({"status":"return error","result":"no_such_taskid"})
    else:
        res =  json.dumps({"status":"running","result":"clusting..."})
    resp = Response(res)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
    
@app.route('/wordcut',methods = ['POST'])
def fenci():
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)

    data = request.get_data()
    data = json.loads(data)
    data = data["sentence"]
    res = fenci_cut(data)
    
    resp = Response(res)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'OPTIONS,HEAD,GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with'
    return resp
    #return res

@app.route('/similarity2sentence',methods = ['POST'])
def similarity2s():
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)

    data = request.get_data()
    try:
        data = json.loads(data)
        s1 = data[0]
        s2 = data[1]

        resp = Response(json.dumps({"score":max(compare(s1,s2, seg=True , version = '2.0'),compare(s2,s1,seg = True,version = '2.0'))}))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
        #return json.dumps({"score":max(compare(s1,s2, seg=True , version = '2.0'),compare(s2,s1,seg = True,version = '2.0'))})
    except Exception as e:
        return str(e)

@app.route('/similarity2word',methods = ['POST'])
def similarity2w():
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)

    data = request.get_data()
    try:
        data = json.loads(data)
        w1 = data[0]
        w2 = data[1]
        
        resp = Response(json.dumps({"score":max(get_word_similar(w1,w2),get_word_similar(w2,w1))}))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
        #return json.dumps({"score":max(get_word_similar(w1,w2),get_word_similar(w2,w1))})
    except Exception as e:
        return str(e)

@app.route('/wordneighbors',methods = ['POST'])
def wordneighbors():
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)

    data = request.get_data()
    try:
        data = json.loads(data)
        word = data["word"]
        n = data["rc"]
        #return word_neighbors(word,n)
        resp = Response(word_neighbors(word,n))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        return str(e)

@app.route('/textrank',methods = ['POST'])
def textranks():
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)

    data = request.get_data()
    try:
        data = json.loads(data)
        text = data["text"]
        res = textrank(text,data)
        resp = Response(res)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
        #return res

    except Exception as e:
        return str(e)

@app.route('/textrank-fast',methods = ['POST'])
def textranks_f():
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)

    data = request.get_data()
    try:
        data = json.loads(data)
        text = data["text"]
        res = textrank_yuan(text,data)
        resp = Response(res)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
        #return res

    except Exception as e:
        return str(e)

@app.route('/wordvec',methods = ['POST'])
def wordvec():
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)
    
    data = request.get_data()
    try:
        data = json.loads(data)
        data = data["word"]
        resp = Response(str(word_model(data)))
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
        #return str(word_model(data))

    except Exception as e:
        return str(e)

@app.route('/word-cloud',methods = ['POST'])
def wordcloud():
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)

    data = request.get_data()
    try:
        data = json.loads(data)
        sentence = data["sentence"]
        topK = data["topK"]
    except Exception as e:
        return str(e)

    resp = Response(word_cloud(sentence = sentence,topK = topK))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
    #return word_cloud(sentence = sentence,topK = topK)

@app.route('/sentence-vec',methods = ['POST'])
def sentence_vec():
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)

    data = request.get_data()
    try:
        data = json.loads(data)
        sentence = data["sentence"]
    except Exception as e:
        return str(e)
    
    resp = Response(sentence_vector(sentence))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
    #return sentence_vector(sentence)

@app.route('/article-emotion',methods = ['POST'])
def article_emotion():
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)

    data = request.get_data()
    try:
        data = json.loads(data)
        title = data["title"]
        post = data["post"]
    except Exception as e:
        return str(e)
    
    resp = Response(predict_article(title = title,post = post))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
    #return predict_article(title = title,post = post)

@app.route('/comment-emotion',methods = ['POST'])
def comment_emotion():
    try:
        token = request.headers["Content-Type"]
        if token != "application/json":
            raise BaseException("invalid type")
    except Exception as e:
        return str(e)

    data = request.get_data()
    try:
        data = json.loads(data)
        comment = data["comment"]
    except Exception as e:
        return str(e)

    
    resp = Response(predict_comment(comment = comment))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp
    #return predict_comment(comment = comment)
    
if __name__ == "__main__":
    app.run(port=xxx,host='0.0.0.0',debug=True) #xxx 就是要启用的端口
