from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
import nltk
nltk.download('punkt')
from math import log, sqrt
import pandas as pd
import numpy as np
import os

def load_files():
    data = []
    for filename in os.listdir('test\\ham'):
        if filename.endswith(".txt"):
            f = open('test\\ham\\'+filename)
            lines = f.read()
            data.append([0, lines])
    for filename in os.listdir('test\\spam'):
        if filename.endswith(".txt"):
            f = open('test\\spam\\'+filename)
            lines = f.read()
            data.append([1, lines])
    return pd.DataFrame(data, columns=['Spam','Text'])

# def process_message(message):


data = load_files()
print(data)
fdist = FreqDist(word_tokenize(data['Text']))
for word, frequency in fdist.most_common(50):
    print(u'{};{}'.format(word, frequency))