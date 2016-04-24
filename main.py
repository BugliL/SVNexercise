import sys
from nltk.corpus import reuters
print "Selezione dei documenti"

common_categories = [
    'earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn'
]
print "Categorie : ",
print common_categories

print "\nInizio selezione dei documenti"
documents = set()
for cat in common_categories:
    d = set(reuters.fileids(cat))
    print "{} : {}".format(cat, len(d))
    documents |= d
print "TOTALI : ",len(documents)

training_documents = filter(lambda doc: doc.startswith(u"train"),documents)
test_documents = filter(lambda doc: doc.startswith(u"test"),documents)

print "\n#Documenti di training : ",
print len(training_documents)

print "#Documenti di test : ",
print len(test_documents)
print "\nFine selezione dei documenti"