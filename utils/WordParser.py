# Natural Language Toolkit: Corpus Readers
#
# Copyright (C) 2001-2016 NLTK Project
# Author: Edward Loper <edloper@gmail.com>
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

# TODO this docstring isn't up-to-date!
"""
NLTK corpus readers.  The modules in this package provide functions
that can be used to read corpus files in a variety of formats.  These
functions can be used to read both the corpus files that are
distributed in the NLTK corpus package, and corpus files that are part
of external corpora.

Available Corpora
=================

Please see http://www.nltk.org/nltk_data/ for a complete list.
Install corpora using nltk.download().

Corpus Reader Functions
=======================
Each corpus module defines one or more "corpus reader functions",
which can be used to read documents from that corpus.  These functions
take an argument, ``item``, which is used to indicate which document
should be read from the corpus:

- If ``item`` is one of the unique identifiers listed in the corpus
  module's ``items`` variable, then the corresponding document will
  be loaded from the NLTK corpus package.
- If ``item`` is a filename, then that file will be read.

Additionally, corpus reader functions can be given lists of item
names; in which case, they will return a concatenation of the
corresponding documents.

Corpus reader functions are named based on the type of information
they return.  Some common examples, and their return types, are:

- words(): list of str
- sents(): list of (list of str)
- paras(): list of (list of (list of str))
- tagged_words(): list of (str,str) tuple
- tagged_sents(): list of (list of (str,str))
- tagged_paras(): list of (list of (list of (str,str)))
- chunked_sents(): list of (Tree w/ (str,str) leaves)
- parsed_sents(): list of (Tree with str leaves)
- parsed_paras(): list of (list of (Tree with str leaves))
- xml(): A single xml ElementTree
- raw(): unprocessed corpus contents

For example, to read a list of the words in the Brown Corpus, use
``nltk.corpus.brown.words()``:

    >>> from nltk.corpus import brown
    >>> print(", ".join(brown.words()))
    The, Fulton, County, Grand, Jury, said, ...

"""

import nltk
import string

from nltk.corpus import reuters
from nltk.corpus import stopwords

class WordParser(object):
    def __init__(self):
        pass

    def removePuctuation(self, word):
        for x in string.punctuation:
            word = word.replace(x, '')
        return word

    def isNumber(self, word):
        """
        Controlla se la parola e' un numero
        :param word: string
        :return: boolean
        """
        try:
            float(self.removePuctuation(word))
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
        for x in string.punctuation:
            word = word.replace(x,'')
        return word == ''

    def removeUselessWordsAndChars(self,txt):
        return " ".join(self.getWords(txt))

    def getWords(self, txt):
        """
        Estrae le parole rilevanti da un testo
        :param txt: string
        :return: list
        """
        tokenizer = nltk.RegexpTokenizer(r'\w+')
        words = tokenizer.tokenize(txt)
        sWords = stopwords.words("english")

        return [w for w in words if w not in sWords and not self.isNumber(w)]

    def getWordsFromReutersDoc(self, doc):
        return self.getWords(reuters.raw(doc))

    def isWord(self,word):
        """
        Controlla se e' una parola
        :param word: string
        :return: boolean
        """
        if ( not word ) or self.isNumber(word) or self.isPunctating(word) or word == ' ':
            return False
        else:
            return True

    def removeStopwords(self, d):
        output = []
        english_stopword = stopwords.words("english")

        for w in d:
            if w not in english_stopword:
                output.append(w)
        return output
