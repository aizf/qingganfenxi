import nltk
import pickle
from nltk.classify import NaiveBayesClassifier
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from build import formatSet


def get_features(each):
    key_words = [
        "JJ", "JJR", "JJS", "MD", "RB", "RBR", "RBS", "UH", "WDT", "WP", "WP$",
        "WRB"
    ]
    # key_words+=["VB","VBD","VBG","VBN","VBP"]
    featureset = {}
    for i in each:
        if i[1] in key_words:
            featureset[i[0]] = True
    return featureset


if __name__ == "__main__":
    lancaster = nltk.LancasterStemmer()
    trainSetPath = r".\text\trainset.txt"
    w=['most',  'it', 'day', 'It', 'Is', 'beauti', "n't", "'s","?","me"]
    print(pos_tag(w))

{"n't", 'It', 'it', 'most'}
{ 'day', 'Is', 'beauti', "n't", "'s"}
