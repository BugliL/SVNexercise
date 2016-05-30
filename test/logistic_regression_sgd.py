
import collections
import logging
import math
import sys

import nltk
import time
from nltk.corpus import reuters, stopwords

# Impostazione dei log
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def loadListFromFile(filename):
    FH = open("../data/"+filename+".txt",'r')
    lines = FH.readlines()
    glines = [line.strip() for line in lines]
    FH.close()
    return glines

def update(W, X, y, t):
    lt = 0.001 #(1 / t)
    a = f(W, X) * y
    xy = (lt * y) * (a < 1)
    for xi in X:
        if not W.has_key(xi):
            W[xi] = xy
        else:
            W[xi] -= lt * W[xi] - xy
    return W


def getCategory(doc):
    # funzione di ritorno per un classificatore binario
    return (reuters.categories(doc)[0] == 'earn' and 1) or 0


def getWords(doc):
    corpus = reuters.raw(doc).lower()
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    words = set(tokenizer.tokenize(corpus))
    return words


def _compareVectors(X1,X2):
    """
    Compara i 2 vettori controllando quale dei 2 sia piu'
    esterno e quale più interno
    :param X1: dict
    :param X2: dict
    :return: int
    """
    pass

def checkData(documentList):
    for doc in documentList:
        y, words = getCategory(doc), getWords(doc)




def training(documentList):
    t = 1
    W = collections.defaultdict(float)
    n = len(documentList)
    p = 100 * t / n
    old_p = 0

    logger.info("Controllo se separabile linearmente".format(100 * t / n))
    checkData(documentList)

    for doc in documentList:
        if p % 10 == 0 and old_p != p:
            old_p = p
            logger.info("Completamento: {:3}%".format(100 * t / n))

        y, words = getCategory(doc), getWords(doc)
        update(W, words, y, float(t))
        t += 1; p = 100 * t / n

    logger.info("Completamento: {:3}%".format(100*t/n))
    return W


def classify(W, X):
    return 1 if 0. < f(W, X) else -1


def f(W, X):
    return sum([W[xi] for xi in X])


def test(W, docs):
    n = 0; m = 0
    for doc in docs:
        c, words = getCategory(doc), getWords(doc)
        l = classify(W, words)
        if l == c:
            m += 1
        n += 1
    print m / float(n)
    return m / float(n)

if __name__ == '__main__':
    # Caricamento del materiale
    logger.info("Caricamento del materiale")
    training_docs = loadListFromFile("training")
    test_docs = loadListFromFile("test")
    categories_docs = loadListFromFile("categories")

    # Impostazione dei parametri iniziali
    N = len(training_docs)
    eta0 = 0.1

    logger.info("Training")
    W = training(training_docs)
    logger.info("Risultato")
    for x, v in W.iteritems():
        if v != 0.0:
            logger.info("{} - {}".format(x, v))
    logger.info("Test")
    test(W, test_docs)
