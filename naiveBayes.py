from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
# import nltk
# nltk.download('punkt') #todo uncomment this and the line above if you don't have punkt downloaded
from math import log, sqrt
import pandas as pd
import numpy as np
import os

def load_files():
    data = []
    hDist = FreqDist()
    sDist = FreqDist()
    hCount = 0
    sCount = 0
    for filename in os.listdir('test\\ham'):
        if filename.endswith(".txt"):
            f = open('test\\ham\\'+filename)
            #lines = word_tokenize(f.read())
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                hDist[word.lower()] += 1
            data.append([0, lines])
            hCount += 1
    for filename in os.listdir('test\\spam'):
        if filename.endswith(".txt"):
            f = open('test\\spam\\'+filename)
            #lines = word_tokenize(f.read())
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                sDist[word.lower()] += 1
            data.append([1, lines])
            sCount += 1
    return pd.DataFrame(data, columns=['Spam','Text']), hDist, sDist, hCount, sCount

def TrainNB():
    data, hDist, sDist, hCount, sCount = load_files()
    hPrior =  hCount / (hCount+sCount)
    sPrior = 1-hPrior
    for word in data['Text']:
        0 #Do calcs here



data, hDist, sDist, hCount, sCount = load_files()
print(data)
#word_tokenize(data['Text'])
for word, frequency in hDist.most_common(50):
    print(u'{};{}'.format(word, frequency))

print(hCount)
print(sCount)