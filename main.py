import io
import pickle
import re
import sys
import json

import nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

from build import trainModal

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class SentimentAnalysis():
    def __init__(self, emos):
        self.emos = emos
        self.loadModals()

    def reTrain(self, emos=[]):
        if emos == []:
            emos = self.emos
        for emo in emos:
            print(emo)
            trainModal.trainModal(emo)
        self.loadModals()

    # 加载模型
    def loadModals(self, emos=[]):
        if emos == []:
            emos = self.emos
        res = []
        for emo in emos:
            classifier_f = open("./modals/" + emo + "Classifier.pickle", "rb")
            classifier = pickle.load(classifier_f)
            classifier_f.close()
            res.append(classifier)
        self.clfs = res
        return res

    def __score(self, text, clfs=[], emos=[]):
        res = {}
        if clfs == []:
            clfs = self.clfs
        if emos == []:
            emos = self.emos
        clfs, emos = self.clfs, self.emos
        pos = ["joy"]
        neg = ["sadness", "anger", "fear"]
        indete = ["surprise"]  # 中性
        features = trainModal.get_features_str(text)
        vaderSIA = SentimentIntensityAnalyzer()
        sScores = vaderSIA.polarity_scores(text)
        for clf, emo in zip(clfs, emos):
            # print(emo)
            emo_prob = clf.prob_classify(features).prob(emo)
            sScoresXprob = float()
            if emo in neg:
                sScoresXprob = sScores['neg'] * 0.7 + emo_prob * 0.3
            elif emo in pos:
                sScoresXprob = sScores['pos'] * 0.7 + emo_prob * 0.3
            else:
                sScoresXprob = abs(sScores['compound']) * 0.7 + emo_prob * 0.3
            res[emo] = sScoresXprob
        return res

    def score(self, text, clfs=[], emos=[]):
        if clfs == []:
            clfs = self.clfs
        if emos == []:
            emos = self.emos
        clfs, emos = self.clfs, self.emos
        pos = ["joy"]
        neg = ["sadness", "anger", "fear"]
        indete = ["surprise"]  # 中性
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
                print("emo_prob*sScores",
                      sScores['neg'] * 0.7 + emo_prob * 0.3)
            elif emo in pos:
                print("emo_prob*sScores",
                      sScores['pos'] * 0.7 + emo_prob * 0.3)
            else:
                print("emo_prob*sScores",
                      abs(sScores['compound']) * 0.7 + emo_prob * 0.3)

    def loadScriptJson(self, path):
        f = open(path, "r")
        self.scriptJson = json.load(f)
        f.close()
        # print(self.scriptJson)

    def byPerScene(self):
        res = []
        for scene in self.scriptJson:
            perSS = []
            for dialog in scene["dialog"]:
                perSS.append(self.__score(dialog["content"]))
            res.append({scene["scene_name"]: perSS})

        f = open(r".\text\ForrestGump_score_byPerScene.json", "w")
        json.dump(res, f)
        f.close()
        return res


# 先前研究显示，人有快乐、悲伤、愤怒、惊讶、恐惧和厌恶6种基本情绪。

if __name__ == "__main__":
    emos = ["joy", "sadness", "anger", "surprise", "fear"]
    sa = SentimentAnalysis(emos)

    # 重新训练模型
    # sa.reTrain()
    # 测试句子
    testText = """He's very smart. """
    sa.score(testText)

    # sa.loadScriptJson(r".\text\ForrestGump_script_id_time_sentiment.json")
    # print(sa.byPerScene())
