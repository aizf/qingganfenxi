# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

stops = stopwords.words("english")


def formatFile(txt_path):
    res = []
    text=""
    f = open(txt_path)
    for line in f:
        # print(line)
        l=line.rstrip().split('##')
        res.append(l)
        text+=(' '+l[0])
    f.close()
    fdist = FreqDist(word.lower() for word in word_tokenize(text))
    return res,fdist

def formatStr(str,customStops=[]):
    """分词，提取词干，标注词性"""
    # lancaster = LancasterStemmer()
    porter = PorterStemmer()
    return pos_tag([porter.stem(w) for w in set(word_tokenize(str)) if w not in customStops])

def formatTaggedSet(taggedSet,customStops=[]):
    """分词，提取词干，标注词性"""
    # lancaster = LancasterStemmer()
    porter = PorterStemmer()
    _taggedSet = [[
        pos_tag([porter.stem(w) for w in set(word_tokenize(i[0])) if w not in customStops]), i[1]
    ] for i in taggedSet]
    return _taggedSet


def getFeatureset(_taggedSet, featuresFun):
    featureset = [(featuresFun(i[0]), i[1]) for i in _taggedSet]
    return featureset


def toFeatureset(taggedSet, featuresFun,customStops):
    return getFeatureset(formatTaggedSet(taggedSet,customStops), featuresFun)
