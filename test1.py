import io
import re
import sys

# import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def get_features(sent):
    _words = set(word_tokenize(sent))
    _stopwords = stopwords.words('english')[:143]
    words = [word.lower() for word in _words if word not in _stopwords]

    # featureList=["is","one"]
    # features={}
    # for word in featureList:
    #     features['contains(%s)' % word]=(word in t_words)
    # return features

    return words


def aaaa():
    print(__name__)


if __name__ == "__main__":
    # aaaa()

    a=1246863422222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222
    print(a**10000)
