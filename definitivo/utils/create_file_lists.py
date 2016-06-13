import sys
from datetime import datetime
from nltk.corpus import reuters
print "Selezione dei documenti"

common_categories = [
    'earn', 'acq', 'money-fx', 'grain', 'crude',
    'trade', 'interest', 'ship', 'wheat', 'corn'
]
print "Categorie : ",
print common_categories

print "\nInizio selezione dei documenti", datetime.now().isoformat()
documents = set()
for cat in common_categories:
    d = set(reuters.fileids(cat))
    print "{} : {}".format(cat, len(d))
    documents |= d
print "TOTALI : ",len(documents)

training_documents = filter(lambda doc: doc.startswith(u"tr"),documents)
test_documents = filter(lambda doc: doc.startswith(u"te"),documents)

print "\n#Documenti di training : ",
print len(training_documents)

print "#Documenti di test : ",
print len(test_documents)

print "\nFine selezione dei documenti", datetime.now().isoformat()

FH = open(u'training.txt','w')
FH.write("\n".join(training_documents))
FH.close()

FH = open(u'test.txt','w')
FH.write("\n".join(test_documents))
FH.close()

FH = open(u'categories.txt','w')
FH.write("\n".join(common_categories))
FH.close()

