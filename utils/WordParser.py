import nltk
import string

class WordParser(object):
    def __init__(self):
        pass

    def isNumber(self, word):
        """
        Controlla se la parola e' un numero
        :param word: string
        :return: boolean
        """
        try:
            float(word) if '.' in word else int(word)
        except ValueError:
            return False
        else:
            return True

    def isPunctating(self,word):
        """
        Controlla se la parola e' punteggiatura
        :param word: string
        :return: boolean
        """
        return len(word) == 1 and word in string.punctuation

    def getWords(self, txt):
        """
        Estrae le parole rilevanti da un testo
        :param txt: string
        :return: list
        """

    def isWord(self,word):
        """
        Controlla se e' una parola
        :param word: string
        :return: boolean
        """
        if self.isNumber(word) or self.isPunctating(word):
            return False
        else:
            return True

