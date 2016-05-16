import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize

txt = """ Compatibility of systems of linear constraints over the set of natural
numbers. Criteria of compatibility of a system of linear Diophantine
equations, strict inequations, and nonstrict inequations are considered.
Upper bounds for components of a minimal set of solutions and algorithms
of construction of minimal generating sets of solutions for all types of
systems are given. These criteria and the corresponding algorithms for
constructing a minimal supporting set of solutions can be used in solving
all the considered types of systems and systems of mixed types. """

def extractWords(text):
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text)
    words = word_tokenize(text)
    sWords = stopwords.words("english")

    return [w.lower() for w in words if w not in sWords]

if __name__ == "__main__":
    print extractWords(txt)