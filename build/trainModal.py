import pickle
import random
import nltk
from nltk.classify import NaiveBayesClassifier
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from . import formatSet


def get_features(each):
    key_words = [
        "JJ", "JJR", "JJS", "MD", "RB", "RBR", "RBS", "UH", "WDT", "WP", "WP$",
        "WRB"
    ]
    key_words += ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    key_words += ["PRP", "CC", "NN", "UH"]
    featureset = {}
    for i in each:
        if i[1] in key_words or i[0] in ["?", "!", "？", "！"]:
            # if True:
            featureset[i[0]] = True
    return featureset


def get_features_str(string, customStops=[]):
    return get_features(formatSet.formatStr(string, customStops))


def trainModal(emo):
    trainSetPath = r".\text\trainset.txt"
    _trainSet, fdist = formatSet.formatFile(trainSetPath)
    fWordsList = [i[0] for i in fdist.most_common(8)]  # 最频繁的词
    random.shuffle(_trainSet)
    # print(trainSet[:2])
    # [['Sometimes our vision clears only after our eyes are washed
    # away with ears. ', 'sadness'], ...]
    trainSet = []
    for i in _trainSet:
        if i[1] == emo:
            trainSet.append([i[0], emo])
        else:
            trainSet.append([i[0], "NOT" + emo])
    # print(trainSet[:10])

    # 分词，提取词干，标注词性，提取特征
    featuresets = formatSet.toFeatureset(trainSet, get_features, fWordsList)
    # print(featuresets[:5])
    divideNum = int(len(featuresets) * 0.80)
    # 划分的数量
    # print("divideNum:", divideNum)

    train_set, test_set = featuresets[:divideNum], featuresets[divideNum:]
    train_ori, test_ori = trainSet[:divideNum], trainSet[divideNum:]

    # vaderSIA = SentimentIntensityAnalyzer()
    # sScores = vaderSIA.polarity_scores(sen)

    classifier = NaiveBayesClassifier.train(train_set)
    # classifier.show_most_informative_features(10)

    print(nltk.classify.accuracy(classifier, test_set))

    # 保存分类器
    save_classifier = open("./modals/" + emo + "Classifier.pickle", "wb")
    pickle.dump(classifier, save_classifier, protocol=pickle.HIGHEST_PROTOCOL)
    save_classifier.close()

    # 导入分类器
    # classifier_f = open("./modals/"+emo+"Classifier.pickle", "rb")
    # classifier = pickle.load(classifier_f)
    # classifier_f.close()


if __name__ == "__main__":
    emos = ["joy", "sadness", "surprise", "anger", "fear"]
    trainModal(emos[0])