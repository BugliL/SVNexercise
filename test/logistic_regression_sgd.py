

"""
    Logistic Regression with Stochastic Gradient Descent.
    Copyright (c) 2009, Naoaki Okazaki

This code illustrates an implementation of logistic regression models
trained by Stochastic Gradient Decent (SGD).

This program reads a training set from STDIN, trains a logistic regression
model, evaluates the model on a test set (given by the first argument) if
specified, and outputs the feature weights to STDOUT. This is the typical
usage of this problem:
    $ ./logistic_regression_sgd.py test.txt < train.txt

Each line in a data set represents an instance that consists of binary
features and label separated by TAB characters. This is the BNF notation
of the data format:

    <line>    ::= <label> ('\t' <feature>)+ '\n'
    <label>   ::= '1' | '0'
    <feature> ::= <string>

The following topics are not covered for simplicity:
    - bias term
    - regularization
    - real-valued features
    - multiclass logistic regression (maximum entropy model)
    - two or more iterations for training
    - calibration of learning rate

This code requires Python 2.5 or later for collections.defaultdict().

"""

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

def update(W, X, l, eta):
    # Compute the inner product of features and their weights.
    a = f(W,X)
    logger.info(" W = {} \n X = {}\n-> a = {} ".format(W.items()[2:10],X,a))
    time.sleep(3)
    # Compute the gradient of the error function (avoiding +Inf overflow).
    g = ((1. / (1. + math.exp(-a))) - l) if -100. < a else (0. - l)

    # Update the feature weights by Stochastic Gradient Descent.
    for x in X:
        W[x] -= eta * g


def getCategory(doc):
    # funzione di ritorno per un classificatore binario
    return (reuters.categories(doc)[0] == 'earn' and 1) or 0


def getWords(doc):
    corpus = reuters.raw(doc).lower()
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    words = set(tokenizer.tokenize(corpus))
    return words


def train(documentList):
    t = 1
    W = collections.defaultdict(float)
    # Loop for instances.
    for doc in documentList:
        logger.info("Documento: " + doc)
        #fields = doc.strip('\n').split('\t')
        c = getCategory(doc)
        words = getWords(doc)
        # update(W, fields[:1], float(fields[0]), eta0 / (1 + t / float(N)))
        update(W, words, float(c), eta0 / (1 + t / float(N)))
        t += 1
    return W

def classify(W, X):
    return 1 if 0. < f(W, X) else 0


def f(W, X):
    return sum([W[x] for x in X])


def test(W, docs):
    n = 0; m = 0
    for doc in docs:
        c = getCategory(doc) # fields = line.strip('\n').split('\t')
        words = getWords(doc)
        l = classify(W, words)
        m += (1 - (l ^ int(c)))
        n += 1
    print('Accuracy = %f (%d/%d)' % (m / float(n), m, n))

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
    W = train(training_docs)
    logger.info("Risultato")
    logger.info(" || W || = {}".format(len(W.keys())))
    # logger.info(W.items())
    logger.info("Test")
    test(W, test_docs)
