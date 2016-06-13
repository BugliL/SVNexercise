
def loadListFromFile(filename):
    FH = open("data/"+filename+".txt",'r')
    lines = FH.readlines()
    glines = [line.strip() for line in lines]
    FH.close()
    return glines
