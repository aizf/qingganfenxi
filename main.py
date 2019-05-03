import io
import pickle
import re
import sys

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

from build import trainModal

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def reTrain(emos):
    for emo in emos:
        trainModal.trainModal(emo)


def loadModals(emos):
    res = []
    for emo in emos:
        classifier_f = open("./modals/" + emo + "Classifier.pickle", "rb")
        classifier = pickle.load(classifier_f)
        classifier_f.close()
        res.append(classifier)
    return res


def score(text, clfs, emos):
    pos = ["joy"]
    neg = ["sadness", "anger", "fear"]
    indete=["surprise"]
    features = trainModal.get_features_str(text)
    print(features)
    vaderSIA = SentimentIntensityAnalyzer()
    sScores = vaderSIA.polarity_scores(text)
    print("sScores", sScores)
    for clf, emo in zip(clfs, emos):
        # print(clf.classify(features))
        print(emo)
        emo_prob = clf.prob_classify(features).prob(emo)
        print("emo_prob", emo_prob)
        if emo in neg:
            print("emo_prob*sScores", sScores['neg'] * emo_prob)
        elif emo in pos:
            print("emo_prob*sScores", sScores['pos'] * emo_prob)
        else:
            print("emo_prob*sScores", abs(sScores['compound']) * emo_prob)


# 先前研究显示，人有快乐、悲伤、愤怒、惊讶、恐惧和厌恶6种基本情绪。

if __name__ == "__main__":
    emos = ["joy", "sadness", "anger", "surprise", "fear"]
    # reTrain(emos)

    clfs = loadModals(emos)

    testText = """The nurse shakes her head, a bit apprehensive about this
strange man next to her."""
    score(testText, clfs, emos)
