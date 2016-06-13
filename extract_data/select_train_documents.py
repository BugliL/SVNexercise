
from nltk.corpus import reuters

train_file_list = [x for x in reuters.fileids() if 'train' in x]
category_list = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn']

count_categories = {}
for c in category_list:
    print("{:8} documenti per {:10}".format(len(reuters.fileids(c)),c))

"""
for f in train_file_list:
    l = lambda cat: cat in category_list

    current_categories = [c for c in reuters.categories(f) if c in category_list]
    if current_categories:
        cat = current_categories[0]
        if cat in count_categories.keys():
            count_categories[cat] += 1
        else:
            count_categories[cat] = 1

for c, v in count_categories.items():
    print ("{:6} documenti di training per la categoria {:8}".format(v, c))

print("Tutte le categorie: ")
for c, v in count_categories.items():
    print ("{:6} documenti di training per la categoria {:8}".format(v, c))
"""
