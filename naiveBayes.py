from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords') #todo uncomment this and the line above if you don't have the stopwords downloaded
stop_words = set(stopwords.words('english'))
from nltk import FreqDist
# import nltk
# nltk.download('punkt') #todo uncomment this and the line above if you don't have punkt downloaded
from math import log
import pandas as pd
import os


def load_files():
    data = []
    hDist = FreqDist()
    sDist = FreqDist()
    hCount = 0
    sCount = 0
    vocabWords = 0
    for filename in os.listdir('train\\ham'):
        if filename.endswith(".txt"):
            f = open('train\\ham\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                hDist[word.lower()] += 1
                if hDist[word.lower()] == 1:
                    vocabWords += 1
            data.append([0, lines])
            hCount += 1
    for filename in os.listdir('train\\spam'):
        if filename.endswith(".txt"):
            f = open('train\\spam\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                sDist[word.lower()] += 1
                if hDist[word.lower()] == 0 and sDist[word.lower()] == 1:
                    vocabWords += 1
            data.append([1, lines])
            sCount += 1
    return pd.DataFrame(data, columns=['Spam','Text']), hDist, sDist, hCount, sCount, vocabWords


def TrainNB():
    data, hDist, sDist, hCount, sCount, vocabWords = load_files()
    hPrior = hCount / (hCount+sCount)
    sPrior = sCount / (sCount+hCount)
    hTotal = 0
    sTotal = 0
    condProbHam = []
    condProbSpam = []
    for word, frequency in hDist.items():
        hTotal += 1
    for word, frequency in sDist.items():
        sTotal += 1
    for word, frequency in hDist.items():
        hamCP = (frequency + 1) / (hTotal + vocabWords)
        condProbHam.append([word, hamCP])
    for word, frequency in sDist.items():
        spamCP = (frequency + 1) / (sTotal + vocabWords)
        condProbSpam.append([word, spamCP])
    hSmooth = log(1 / (hTotal + vocabWords))
    sSmooth = log(1 / (sTotal + vocabWords))
    return condProbHam, condProbSpam, hPrior, sPrior, hSmooth, sSmooth


def applyNB(hPrior, sPrior, condProbHam, condProbSpam, hSmooth, sSmooth):
    condProbHam = pd.DataFrame(condProbHam, columns=['Text', 'CondProb'])
    condProbSpam = pd.DataFrame(condProbSpam, columns=['Text', 'CondProb'])
    totalFiles = 0
    correct = 0
    for filename in os.listdir("test\\ham"):
        hScore = log(hPrior)
        sScore = log(sPrior)
        if filename.endswith(".txt"):
            f = open('test\\ham\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                word = word.lower()
                temp = condProbHam.loc[condProbHam.Text == word]
                try:
                    hScore += log(temp.iloc[0, 1])
                except IndexError:
                    hScore += hSmooth
                temp = condProbSpam.loc[condProbSpam.Text == word]
                try:
                    sScore += log(temp.iloc[0, 1])
                except IndexError:
                    sScore += sSmooth
            if hScore >= sScore:
                correct += 1
            totalFiles += 1

    for filename in os.listdir("test\\spam"):
        hScore = log(hPrior)
        sScore = log(sPrior)
        if filename.endswith(".txt"):
            f = open('test\\spam\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                word = word.lower()
                temp = condProbHam.loc[condProbHam.Text == word]
                try:
                    hScore += log(temp.iloc[0, 1])
                except IndexError:
                    hScore += hSmooth
                temp = condProbSpam.loc[condProbSpam.Text == word]
                try:
                    sScore += log(temp.iloc[0, 1])
                except IndexError:
                    sScore += sSmooth
            if sScore >= hScore:
                correct += 1
            totalFiles += 1
    return (correct/totalFiles)


def load_filesStopwords():
    data = []
    hDist = FreqDist()
    sDist = FreqDist()
    hCount = 0
    sCount = 0
    vocabWords = 0
    for filename in os.listdir('train\\ham'):
        if filename.endswith(".txt"):
            f = open('train\\ham\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if
                     word.isalnum() and word.lower() not in stop_words]
            for word in lines:
                hDist[word.lower()] += 1
                if hDist[word.lower()] == 1:
                    vocabWords += 1
            data.append([0, lines])
            hCount += 1
    for filename in os.listdir('train\\spam'):
        if filename.endswith(".txt"):
            f = open('train\\spam\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if
                     word.isalnum() and word.lower() not in stop_words]
            for word in lines:
                sDist[word.lower()] += 1
                if hDist[word.lower()] == 0 and sDist[word.lower()] == 1:
                    vocabWords += 1
            data.append([1, lines])
            sCount += 1
    return pd.DataFrame(data, columns=['Spam','Text']), hDist, sDist, hCount, sCount, vocabWords


def TrainNBStopwords():
    data, hDist, sDist, hCount, sCount, vocabWords = load_filesStopwords()
    hPrior = hCount / (hCount+sCount)
    sPrior = sCount / (sCount+hCount)
    hTotal = 0
    sTotal = 0
    condProbHam = []
    condProbSpam = []
    for word, frequency in hDist.items():
        hTotal += 1
    for word, frequency in sDist.items():
        sTotal += 1
    for word, frequency in hDist.items():
        hamCP = (frequency + 1) / (hTotal + vocabWords)
        condProbHam.append([word, hamCP])
    for word, frequency in sDist.items():
        spamCP = (frequency + 1) / (sTotal + vocabWords)
        condProbSpam.append([word, spamCP])
    hSmooth = log(1 / (hTotal + vocabWords))
    sSmooth = log(1 / (sTotal + vocabWords))
    return condProbHam, condProbSpam, hPrior, sPrior, hSmooth, sSmooth


def applyNBStopwords(hPrior, sPrior, condProbHam, condProbSpam, hSmooth, sSmooth):
    condProbHam = pd.DataFrame(condProbHam, columns=['Text', 'CondProb'])
    condProbSpam = pd.DataFrame(condProbSpam, columns=['Text', 'CondProb'])
    totalFiles = 0
    correct = 0
    for filename in os.listdir("test\\ham"):
        hScore = log(hPrior)
        sScore = log(sPrior)
        if filename.endswith(".txt"):
            f = open('test\\ham\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if
                     word.isalnum() and word.lower() not in stop_words]
            for word in lines:
                word = word.lower()
                temp = condProbHam.loc[condProbHam.Text == word]
                try:
                    hScore += log(temp.iloc[0, 1])
                except IndexError:
                    hScore += hSmooth
                temp = condProbSpam.loc[condProbSpam.Text == word]
                try:
                    sScore += log(temp.iloc[0, 1])
                except IndexError:
                    sScore += sSmooth
            if hScore > sScore:
                correct += 1
            totalFiles += 1

    for filename in os.listdir("test\\spam"):
        hScore = log(hPrior)
        sScore = log(sPrior)
        if filename.endswith(".txt"):
            f = open('test\\spam\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum() and word.lower() not in stop_words]
            #lines = [w for w in lines if w not in stop_words]
            for word in lines:
                word = word.lower()
                temp = condProbHam.loc[condProbHam.Text == word]
                try:
                    hScore += log(temp.iloc[0, 1])
                except IndexError:
                    hScore += hSmooth
                temp = condProbSpam.loc[condProbSpam.Text == word]
                try:
                    sScore += log(temp.iloc[0, 1])
                except IndexError:
                    sScore += sSmooth
            if sScore > hScore:
                correct += 1
            totalFiles += 1
    return (correct/totalFiles)


def main():
    condProbHam, condProbSpam, hPrior, sPrior, hSmooth, sSmooth = TrainNB()
    nbC = applyNB(hPrior, sPrior, condProbHam, condProbSpam, hSmooth, sSmooth)
    print("NB correctness: ")
    print(nbC)
    condProbHam, condProbSpam, hPrior, sPrior, hSmooth, sSmooth = TrainNBStopwords()
    nbC = applyNBStopwords(hPrior, sPrior, condProbHam, condProbSpam, hSmooth, sSmooth)
    print("NB correctness w/o stopwords: ")
    print(nbC)

if __name__== "__main__":
  main()