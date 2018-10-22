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
    for filename in os.listdir('test\\ham'):
        if filename.endswith(".txt"):
            f = open('test\\ham\\' + filename)
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
    for filename in os.listdir('test\\spam'):
        if filename.endswith(".txt"):
            f = open('test\\spam\\' + filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                word = word.lower()
                if word not in sDic:
                    sDic[word] = count
                    count += 1
    return hDic, sDic


def createDictionaryStopwords():
    hDic = {}
    count = 0
    for filename in os.listdir('train\\ham'):
        if filename.endswith(".txt"):
            f = open('train\\ham\\'+filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum() and word.lower() not in stop_words]
            for word in lines:
                word = word.lower()
                if word not in hDic:
                    hDic[word] = count
                    count += 1
    for filename in os.listdir('test\\ham'):
        if filename.endswith(".txt"):
            f = open('test\\ham\\' + filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum() and word.lower() not in stop_words]
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
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum() and word.lower() not in stop_words]
            for word in lines:
                word = word.lower()
                if word not in sDic:
                    sDic[word] = count
                    count += 1
    for filename in os.listdir('test\\spam'):
        if filename.endswith(".txt"):
            f = open('test\\spam\\' + filename)
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum() and word.lower() not in stop_words]
            for word in lines:
                word = word.lower()
                if word not in sDic:
                    sDic[word] = count
                    count += 1
    return hDic, sDic

# def createDictionaryTest():
#     hDic = {}
#     count = 0
#     for filename in os.listdir('test\\ham'):
#         if filename.endswith(".txt"):
#             f = open('test\\ham\\'+filename)
#             lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
#             for word in lines:
#                 word = word.lower()
#                 if word not in hDic:
#                     hDic[word] = count
#                     count += 1
#     sDic = {}
#     count = 0
#     for filename in os.listdir('test\\spam'):
#         if filename.endswith(".txt"):
#             f = open('test\\spam\\'+filename)
#             lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
#             for word in lines:
#                 word = word.lower()
#                 if word not in sDic:
#                     sDic[word] = count
#                     count += 1
#     return hDic, sDic


# def vectorize(hDic, sDic):
#     hVectors = []
#     for filename in os.listdir('train\\ham'):
#         if filename.endswith(".txt"):
#             vector = [0] * len(hDic)
#             f = open('train\\ham\\'+filename)
#             lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
#             for word in lines:
#                 word = word.lower()
#                 index = hDic[word]
#                 vector[index] += 1
#             hVectors.append(vector)
#     sVectors = []
#     for filename in os.listdir('train\\spam'):
#         if filename.endswith(".txt"):
#             vector = [0] * len(sDic)
#             f = open('train\\spam\\'+filename)
#             lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
#             for word in lines:
#                 word = word.lower()
#                 index = sDic[word]
#                 vector[index] += 1
#             sVectors.append(vector)
#     return np.array(hVectors, sVectors)


def vectorize(hDic, sDic):
    vectors = []
    y = []
    for filename in os.listdir('train\\ham'):
        if filename.endswith(".txt"):
            f = open('train\\ham\\'+filename)
            vector = [0] * (len(hDic) + len(sDic))
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                word = word.lower()
                index = hDic[word]
                vector[index] += 1
            vectors.append(vector)
            y.append(0)
    for filename in os.listdir('train\\spam'):
        if filename.endswith(".txt"):
            f = open('train\\spam\\'+filename)
            vector = [0] * (len(hDic) + len(sDic))
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                word = word.lower()
                index = sDic[word]
                vector[index] += 1
            vectors.append(vector)
            y.append(1)
    return np.array(vectors), y


def vectorizeStopwords(hDic, sDic):
    vectors = []
    y = []
    for filename in os.listdir('train\\ham'):
        if filename.endswith(".txt"):
            f = open('train\\ham\\'+filename)
            vector = [0] * (len(hDic) + len(sDic))
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum() and word.lower() not in stop_words]
            for word in lines:
                word = word.lower()
                index = hDic[word]
                vector[index] += 1
            vectors.append(vector)
            y.append(0)
    for filename in os.listdir('train\\spam'):
        if filename.endswith(".txt"):
            f = open('train\\spam\\'+filename)
            vector = [0] * (len(hDic) + len(sDic))
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum() and word.lower() not in stop_words]
            for word in lines:
                word = word.lower()
                index = sDic[word]
                vector[index] += 1
            vectors.append(vector)
            y.append(1)
    return np.array(vectors), y


def vectorizeTest(hDic, sDic):
    vectors = []
    y = []
    for filename in os.listdir('test\\ham'):
        if filename.endswith(".txt"):
            f = open('test\\ham\\'+filename)
            vector = [0] * (len(hDic) + len(sDic))
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                word = word.lower()
                index = hDic[word]
                vector[index] += 1
            vectors.append(vector)
            y.append(0)
    for filename in os.listdir('test\\spam'):
        if filename.endswith(".txt"):
            f = open('test\\spam\\'+filename)
            vector = [0] * (len(hDic) + len(sDic))
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum()]
            for word in lines:
                word = word.lower()
                index = sDic[word]
                vector[index] += 1
            vectors.append(vector)
            y.append(1)
    return np.array(vectors), y


def vectorizeTestStopwords(hDic, sDic):
    vectors = []
    y = []
    for filename in os.listdir('test\\ham'):
        if filename.endswith(".txt"):
            f = open('test\\ham\\'+filename)
            vector = [0] * (len(hDic) + len(sDic))
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum() and word.lower() not in stop_words]
            for word in lines:
                word = word.lower()
                index = hDic[word]
                vector[index] += 1
            vectors.append(vector)
            y.append(0)
    for filename in os.listdir('test\\spam'):
        if filename.endswith(".txt"):
            f = open('test\\spam\\'+filename)
            vector = [0] * (len(hDic) + len(sDic))
            lines = [word for sent in sent_tokenize(f.read()) for word in word_tokenize(sent) if word.isalnum() and word.lower() not in stop_words]
            for word in lines:
                word = word.lower()
                index = sDic[word]
                vector[index] += 1
            vectors.append(vector)
            y.append(1)
    return np.array(vectors), y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def train(X,y, num_iter, lr, lbda):
    theta = np.zeros(X.shape[1])

    for i in range(num_iter):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / len(y)
        theta -= lr * gradient + (lbda * lr * theta)

        z = np.dot(X, theta)
        h = sigmoid(z)
        #lossV = loss(h, y)
    return theta


def predict(X, theta):
    #X = addIntercept(X)
    prob = sigmoid(np.dot(X, theta))
    return prob >= .5


def evaluate(testX, testY, theta):
    yPredicted = predict(testX, theta)
    correct = 0
    for i,y in enumerate(testY):
        if y == 0:
            y = False
        else:
            y = True
        if y == yPredicted[i]:
            correct += 1
    total = yPredicted.size
    return (correct/total)



def main():
    print("With Stopwords")

    hDic, sDic = createDictionary()
    X, y = vectorize(hDic, sDic)
    theta = train(X, y, 1000, .1, .1)
    X, y = vectorizeTest(hDic, sDic)
    print("iter: 1000, learning rate: .1, lambda: .1")
    print(evaluate(X, y, theta))

    hDic, sDic = createDictionary()
    X, y = vectorize(hDic, sDic)
    theta = train(X, y, 100, .1, .1)
    X, y = vectorizeTest(hDic, sDic)
    print("iter: 100, learning rate: .1, lambda: .1")
    print(evaluate(X, y, theta))

    hDic, sDic = createDictionary()
    X, y = vectorize(hDic, sDic)
    theta = train(X, y, 1000, .2, .1)
    X, y = vectorizeTest(hDic, sDic)
    print("iter: 1000, learning rate: .2, lambda: .1")
    print(evaluate(X, y, theta))

    hDic, sDic = createDictionary()
    X, y = vectorize(hDic, sDic)
    theta = train(X, y, 1000, .1, .2)
    X, y = vectorizeTest(hDic, sDic)
    print("iter: 1000, learning rate: .1, lambda: .2")
    print(evaluate(X, y, theta))

    hDic, sDic = createDictionary()
    X, y = vectorize(hDic, sDic)
    theta = train(X, y, 100, .2, .2)
    X, y = vectorizeTest(hDic, sDic)
    print("iter: 100, learning rate: .2, lambda: .2")
    print(evaluate(X, y, theta))

    print("Without Stopwords")

    hDic, sDic = createDictionaryStopwords()
    X, y = vectorizeStopwords(hDic, sDic)
    theta = train(X, y, 1000, .1, .1)
    X, y = vectorizeTestStopwords(hDic, sDic)
    print("iter: 1000, learning rate: .1, lambda: .1")
    print(evaluate(X, y, theta))

    hDic, sDic = createDictionaryStopwords()
    X, y = vectorizeStopwords(hDic, sDic)
    theta = train(X, y, 100, .1, .1)
    X, y = vectorizeTestStopwords(hDic, sDic)
    print("iter: 100, learning rate: .1, lambda: .1")
    print(evaluate(X, y, theta))

    hDic, sDic = createDictionaryStopwords()
    X, y = vectorizeStopwords(hDic, sDic)
    theta = train(X, y, 1000, .2, .1)
    X, y = vectorizeTestStopwords(hDic, sDic)
    print("iter: 1000, learning rate: .2, lambda: .1")
    print(evaluate(X, y, theta))

    hDic, sDic = createDictionaryStopwords()
    X, y = vectorizeStopwords(hDic, sDic)
    theta = train(X, y, 1000, .1, .2)
    X, y = vectorizeTestStopwords(hDic, sDic)
    print("iter: 1000, learning rate: .1, lambda: .2")
    print(evaluate(X, y, theta))

    hDic, sDic = createDictionaryStopwords()
    X, y = vectorizeStopwords(hDic, sDic)
    theta = train(X, y, 100, .2, .2)
    X, y = vectorizeTestStopwords(hDic, sDic)
    print("iter: 100, learning rate: .2, lambda: .2")
    print(evaluate(X, y, theta))

if __name__== "__main__":
  main()