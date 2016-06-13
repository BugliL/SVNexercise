# coding=utf-8
import logging
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

import numpy as np
from sklearn import linear_model, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import sys
from nltk.corpus import reuters
import matplotlib.pyplot as plt
from SGD_SVM_2 import SGD_SVM
from utils import file as futils


def getSVMCategory(category):
    return -1 if category == 3 else 1

def print_category_info(categories):

    cat_num = {}; i = 1
    for c in categories:
        cat_num[c] = i
        i += 1

    for c,i in cat_num.items():
        logger.info("La categoria {:10} e' {:3}".format(c,getSVMCategory(i)))

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

    cat_num = {}; i = 1
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

    return y, corpus

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
    logger.debug("{:-8} categorie diverse".format(len(categories_docs)))
    logger.debug(str(categories_docs))
    print_category_info(categories_docs)


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
    svm.fit(matrix=train_matrix, categories=train_categories)
    logger.info("Prediction.....")

    # verifica dei risultati
    # hit rate - validita' delle previsioni in termini di percentuale
    logger.info("Metriche del calcolo")
    accuracy, y_true, y_scores = svm.score(test_matrix, test_categories)
    # logger.info("y_true = {}".format(y_true[:100]))
    # logger.info("y_scores = {}".format(y_scores[:100]))

    logger.info("y_true : {}".format(y_true))
    logger.info("y_scores : {}".format(y_scores))

    fpr, tpr, thresholds = roc_curve(np.array(y_true), np.array(y_scores))
    logger.info("fpr : {}".format(fpr))
    logger.info("tpr : {}".format(tpr))
    logger.info("thresholds : {}".format(thresholds))

    roc_auc = metrics.auc(fpr, tpr)
    logger.info("AUC : {}".format(roc_auc))
    logger.info("Accuracy : {}".format(accuracy))

    # Plot of a ROC curve for a specific class
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    # La confusion matrix non ha molto senso con queste dimensioni
    # confusion matrix - matrice che riporta
    # logger.info("CONFUSION MATRIX")
    # print(confusion_matrix(predictions, test_categories))
