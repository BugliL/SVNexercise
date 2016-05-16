# coding=utf-8
from nltk.corpus import reuters

from utils.WordParser import WordParser


def read_file(filename):
    """
        Importa i dat da tutti i file
    :return: str
    """
    FH = open(filename, 'r')
    lines = list(line.strip() for line in FH.readlines())
    FH.close()
    return lines


def read_datas():
    """
    Preleva l'elenco dei file di test e di training
    e le categorie a cui appartengono
    Dati creati con una pre elaborazione con create_file_lists
    sulle 10 categorie principali
    :return: categorie, doc test, doc training
    """
    categories = read_file('categories.txt')
    test_doc = read_file('test.txt')
    training_doc = read_file('training.txt')
    return categories, test_doc, training_doc

categories, test_doc, training_doc = read_datas()
wordP = WordParser()
for w in reuters.words(training_doc[14].strip()):
    if (wordP.isWord(w)):
        print w

#matrix = create_occurrency_matrix(training_doc)