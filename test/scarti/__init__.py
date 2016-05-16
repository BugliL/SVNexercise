import nltk
from nltk.stem.snowball import EnglishStemmer

stemmer = EnglishStemmer()
def tokenize(text):
    tokenizer = nltk.RegexpTokenizer('[a-z]\w+')
    tokens = tokenizer.tokenize(text)
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed