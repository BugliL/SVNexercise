# coding=utf-8
import logging
import random
from sklearn.metrics import roc_curve
from sklearn.svm import SVC

import numpy as np
from sklearn import metrics, linear_model
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import reuters
import matplotlib.pyplot as plt
from SVM import SGD_SVM
from utils import file as futils



def format_data(docs, all_categories):
    y = []; corpus = []
    for d in docs:
        current_categories = filter(lambda x: x in all_categories,reuters.categories(d))
        if current_categories:
            y.append(current_categories[0])
            corpus.append(reuters.raw(d).lower())
    return y, corpus


def set_project_logger():
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    return logger


def plot_results(fpr, tpr, roc_auc, cat='sample'):
    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = {:.3}) [{}]'.format(roc_auc, cat))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('./output/'+cat+'.png')

def filter_docs_with_categories(x):
    categories_docs = futils.loadListFromFile("categories")
    for c in reuters.categories(x):
        if c.lower() in categories_docs:
            return True
    return False

if __name__ == "__main__":
    """
    Lista dei file reuters caccalacca
      Creazione dei dati di addestramento
      Vettorizzazioned dei dati di addestramento - Bag of words
        1) creazione dei Token
        2) assegnazione di un intero ad ogni token
        3) conta dei token in un documento
        4) normalizzazione
        5) creazione di matrice
      Creazione di SVM e il suo addestramento
      Predizioni e test
    """

    logger = set_project_logger()

    # -------------------- Lettura lista dei documenti e categorie ----------------
    logger.info("Lista dei documenti reuters")
    categories_docs = futils.loadListFromFile("categories")
    docs = filter(filter_docs_with_categories, reuters.fileids())

    training_docs = [x for x in docs if 'training' in x]
    test_docs = [x for x in docs if 'test' in x]

    logger.debug("{:-8} documenti per il training".format(len(training_docs)))
    logger.debug("{:-8} documenti per il test".format(len(test_docs)))
    logger.debug("{:-8} categorie diverse".format(len(categories_docs)))
    logger.debug(str(categories_docs))

    train_corpus = [reuters.raw(f).lower() for f in training_docs]
    test_corpus = [reuters.raw(f).lower() for f in test_docs]
    # -----------------------------------------------------------------------------

    # --------------------- Vettorizzazione dei documenti ----------------------------------
    logger.info("Vettorizzazione dei documenti")
    vectorizer = CountVectorizer(min_df=1, stop_words='english', analyzer='word', token_pattern='[a-z]\w+')
    train_matrix = vectorizer.fit_transform(train_corpus).toarray()
    test_matrix = vectorizer.transform(test_corpus).toarray()
    vocab = vectorizer.get_feature_names()
    n = len(vocab)
    logger.info("Numero di parole nella bag of words : {}".format(n))
    # ---------------------------------------------------------------------------------------

    # Per ogni categoria eseguo l'analisi
    for current_cat in categories_docs:

        # --------------------- Selezione delle categorie ----------------------------------
        logger.info("Categoria -1 : {}".format(current_cat))
        train_categories = [-1 if current_cat in reuters.categories(f) else 1 for f in training_docs]
        test_categories = [-1 if current_cat in reuters.categories(f) else 1 for f in test_docs]
        # --------------------------------------------------------------------------------------

        # --------------------- Creazione SVM e addestramento ----------------------------------
        logger.info("Creazione SVM")
        svm = SGD_SVM()

        logger.info("Training SVM...")
        svm.fit(matrix=train_matrix, categories=train_categories)
        # --------------------------------------------------------------------------------------

        # --------------------- Prediction e metriche ---------------------------------------------
        logger.info("Prediction.....")
        logger.info("Metriche del calcolo")

        accuracy, y_true, y_scores = svm.score(test_matrix, test_categories)
        fpr, tpr, thresholds = roc_curve(np.array(y_true), np.array(y_scores))
        roc_auc = metrics.auc(fpr, tpr)

        #accuracy2 = svm2.score(test_matrix, test_categories_evalueted)

        logger.info("AUC : {}".format(roc_auc))
        logger.info("Accuracy : {}".format(accuracy))
        #logger.info("Reference Accuracy : {}".format(accuracy2))

        plot_results(fpr, tpr, roc_auc, current_cat)

