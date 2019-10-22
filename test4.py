import bs4 as bs
import urllib.request
import re
import nltk
import heapq
import json
import numpy as np
from flask import Flask, render_template, redirect, url_for, request
from flask import make_response
from flask import request, Response

app = Flask(__name__)

nltk.download('stopwords')
nltk.download('punkt')


class MyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyJsonEncoder, self).default(obj)


# print(json.dumps(json_1, cls=MyJsonEncoder))


# localhost:5500/zhaiyao
@app.route("/zhaiyao", methods=['POST'])
def zhaiyao():
    # with open("新建文本文档.txt",'r') as f:
    #     txt = f.read()
    # 获取post数据  json.loads(request.form['***'])

    txt = request.form['shuju']
    text = re.sub(r'\[[0-9]*\]', ' ', txt)
    text = re.sub(r'\s+', ' ', text)
    clean_text = text.lower()
    clean_text = re.sub(r'\w', ' ', clean_text)
    clean_text = re.sub(r'\d', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    sentences = nltk.sent_tokenize(text)
    stop_words = nltk.corpus.stopwords.words('english')
    print(clean_text)

    word2count = {}
    for word in nltk.word_tokenize(clean_text):
        if word not in stop_words:
            if word not in word2count.keys():
                word2count[word] = 1
            else:
                word2count[word] += 1

    for key in word2count.keys():
        word2count[key] = word2count[key] / max(word2count.values())

    sent2score = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word2count.keys():
                if len(sentence.split(' ')) < 30:
                    if sentence not in sent2score.keys():
                        sent2score[sentence] = word2count[word]
                    else:
                        sent2score[sentence] += word2count[word]

    best_sentences = heapq.nlargest(5, sent2score, key=sent2score.get)
    result = []
    xuhao = 1
    for sentence in best_sentences:
        result.append(str(xuhao) + ":" + sentence + "<br/>")
        xuhao = xuhao + 1
    stri = json.dumps(result)
    resp = make_response(stri)
    resp.headers['Content-Type'] = "application/json"
    resp.headers['Access-Control-Allow-Origin'] = "*"
    return resp


def main():
    app.run(host='0.0.0.0', port=5500)


if __name__ == "__main__":
    main()