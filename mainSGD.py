# coding=utf-8
import logging
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import sys
from nltk.corpus import reuters

from SGD_SVM import SGD_SVM
from utils import file as futils


def getSVMCategory(category):
    return -1 if category == 3 else 1

def create_tfidf_data(docs,categories,n=None):
    """
    Crea una struttura [(label,[parole])] parsando il documento
    :param docs: lista dei documenti reuters
    :param categories: nomi delle categorie da considerare
    :param n: numero di documenti da usare
    :return: list
    """
    if n:
        docs = docs[:n]

    cat_num = {}
    i = 1
    for c in categories:
        cat_num[c] = i
        i += 1

    y = []
    corpus = []
    for d in docs:
        c = reuters.categories(d)[0]
        if c in categories:
            y.append(getSVMCategory(cat_num[c]))
            corpus.append(reuters.raw(d).lower())
    return y[:1000], corpus[:1000]

if __name__ == "__main__":
    """
    Lista dei file reuters
      Creazione dei dati di addestramento
      Vettorizzazioned dei dati di addestramento - Bag of words
        1) creazione dei Token
        2) assegnazione di un intero ad ogni token
        3) conta dei token in un documento
        4) normalizzazione
        5) creazione di matrice
      Creazione di SVM e il suo addestramento
      Fare delle predizioni
    """
    # Impostazione dei log
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Lettura dei documenti
    logger.info("Lista dei documenti reuters")
    training_docs = futils.loadListFromFile("training")
    test_docs = futils.loadListFromFile("test")
    categories_docs = futils.loadListFromFile("categories")

    logger.debug("{:-8} documenti per il training".format(len(training_docs)))
    logger.debug("{:-8} documenti per il test".format(len(test_docs)))
    logger.debug("{:-8} categoridef getCategory(self, category)".format(len(categories_docs)))

    # struttura [ (categoria, [parole]), (categoria, [parole]), ...]
    logger.info("Caricamento dati da processare")
    train_categories, train_corpus = create_tfidf_data(training_docs,categories_docs)
    test_categories, test_corpus = create_tfidf_data(test_docs,categories_docs)

    # Vettorizzazione
    logger.info("Vettorizzazione dei documenti")
    # -----------------------------------------------------------------------------------------------------------
    # vectorizer = TfidfVectorizer(min_df=1, stop_words='english', analyzer='word')
    # vectorizer = linear_model.SGDClassifier()

    vectorizer = CountVectorizer(min_df=1, stop_words='english',analyzer='word',token_pattern='[a-z]\w+')
    train_matrix = vectorizer.fit_transform(train_corpus).toarray()
    test_matrix = vectorizer.transform(test_corpus).toarray()

    # finita questa fase ho trasformato tutti i documenti in vettori
    # -----------------------------------------------------------------------------------------------------------

    vocab = vectorizer.get_feature_names()
    n = len(vocab)
    logger.info("Numero di parole nella bag of words : {}".format(n))

    logger.info("Creazione SVM")
    svm = SGD_SVM()

    # svm = SVC(C=1000000.0, gamma='auto', kernel='rbf')
    # svm = SGD_SVM(n)
    logger.info("Training SVM...")
    svm.fit(matrix=train_matrix, categories=train_categories, n=n)
    logger.info("Prediction.....")

    # verifica dei risultati
    # hit rate - validita' delle previsioni in termini di percentuale
    logger.info("Calcolo di HIT RATE")
    logger.info("Hit rate : {}".format(svm.score(test_matrix, test_categories)))

    # La confusion matrix non ha molto senso con queste dimensioni
    # confusion matrix - matrice che riporta
    # logger.info("CONFUSION MATRIX")
    # print(confusion_matrix(predictions, test_categories))
