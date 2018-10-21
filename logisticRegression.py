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
import numpy as np
import os

def createDictionary():
    hDic = {}
    count = 0
    for filename in os.listdir('train\\ham'):
        if filename.endswith(".txt"):
            f = open('train\\ham\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                word = word.lower()
                if word not in hDic:
                    hDic[word] = count
                    count += 1
    sDic = {}
    count = 0
    for filename in os.listdir('train\\spam'):
        if filename.endswith(".txt"):
            f = open('train\\spam\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                word = word.lower()
                if word not in sDic:
                    sDic[word] = count
                    count += 1
    return hDic, sDic


def vectorize(hDic, sDic):
    hVectors = []
    for filename in os.listdir('train\\ham'):
        if filename.endswith(".txt"):
            vector = [0] * len(hDic)
            f = open('train\\ham\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                word = word.lower()
                index = hDic[word]
                vector[index] += 1
            hVectors.append(vector)
    sVectors = []
    for filename in os.listdir('train\\spam'):
        if filename.endswith(".txt"):
            vector = [0] * len(sDic)
            f = open('train\\spam\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                word = word.lower()
                index = sDic[word]
                vector[index] += 1
            sVectors.append(vector)
    return np.array(hVectors, sVectors)


def addIntercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def train(X,y):
    X = addIntercept(X)
# weights initialization
    theta = np.zeros(X.shape[1])
    print(theta)

# gradiet descent
    for i in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        theta -= lr * gradient

        z = np.dot(X, theta)
        h = sigmoid(z)
        loss = loss(h, y)


def predict(X): #todo needs to be modified for your class situation
    file = open("logistic_regression_model.pickle",'rb')
    theta = pickle.load(file)
    X = addIntercept(X)
    prob = sigmoid(np.dot(X, theta))
    return prob >= threshold


 def evaluate(testX, testY): #todo needs to be modified for class situation
        yPredicted = predict(testX)
        correct = 0
        for i,y in enumerate(testY):
            if y == 0:
                y = False
            else:
                y = True
            if y == yPredicted[i]:
                correct = correct + 1
        total = yPredicted.size

        return (correct/total)*100


hDic, sDic = createDictionary()
hVectors, sVectors = vectorize(hDic, sDic)
