# coding=utf-8
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from nltk.corpus import reuters

from utils import file as futils


def create_tfidf_data(docs,n=None):
    """
    Crea una struttura [(label,[parole])] togliendo le stopwords
    e parsando il documento
    :param docs: lista dei documenti reuters
    :param n: numero di documenti da usare
    :return: list
    """
    if n:
        docs = docs[:n]
    y = [reuters.categories(d)[0] for d in docs]
    corpus = [reuters.raw(d).lower() for d in docs]
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
    logger.info("Lettura dei documenti reuters")
    training_docs = futils.loadListFromFile("training")
    test_docs = futils.loadListFromFile("test")
    categories_docs = futils.loadListFromFile("categories")

    logger.debug("{:-8} documenti per il training".format(len(training_docs)))
    logger.debug("{:-8} documenti per il test".format(len(test_docs)))
    logger.debug("{:-8} categorie diverse".format(len(categories_docs)))

    # struttura [ (categoria, [parole]), (categoria, [parole]), ...]
    logger.info("Preparazione dati da processare")
    train_categories, train_corpus = create_tfidf_data(training_docs)
    test_categories, test_corpus = create_tfidf_data(test_docs)

    # Vettorizzazione
    # vectorizer = CountVectorizer(min_df=1, stop_words='english',analyzer='word',token_pattern='[a-z]\w+')
    logger.info("Vettorizzazione")
    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', analyzer='word')
    train_matrix = vectorizer.fit_transform(train_corpus)
    test_matrix = vectorizer.transform(test_corpus)

    # matrix e' una matrice di documenti-occorrenze_token
    # categories contiene le categorie di ogni documento
    # (train_matrix, train_categories)

    # support vector classifier
    logger.info("Creazione SVM")
    svm = SVC(C=1000000.0, gamma='auto', kernel='rbf')
    logger.info("Training...")
    svm.fit(train_matrix, train_categories)
    logger.info("Prediction.....")
    predictions = svm.predict(test_matrix)

    # verifica dei risultati
    # hit rate - validita' delle previsioni in termini di percentuale
    logger.info("HIT RATE")
    print(svm.score(test_matrix, test_categories))

    # La confusion matrix non ha molto senso con queste dimensioni
    # confusion matrix - matrice che riporta
    # logger.info("CONFUSION MATRIX")
    # print(confusion_matrix(predictions, test_categories))
