import numpy as np
import abc
from scipy import sparse


threshold = 2


class WordAxisFeaturesABC:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        return

    def getFeatureCount(self):
        return self.getWordCount() * 2

    # token zahl (55000) entweder übergeben oder aus wordlist
    @abc.abstractmethod
    def getWordCount(self):
        return

    # id aus wortliste oder document
    # integer id oder -1 wenn unwichtig
    @abc.abstractmethod
    def getTokenIds(self, page):
        return

    # jedes wort, id der Wörter auf Achse (als numpy array, pro Wort 2 Spalten (links top)

    def extractFeatures(self, page, filename=None):

        tokenIds = self.getTokenIds(page)
        resultNPArray = np.zeros((len(page.left), self.getWordCount() * 2), dtype=np.int8)

        numpyarray = page.as_matrix(["left", "right", "top", "bottom"])

        left_numpy = 0
        right_numpy = 1
        top_numpy = 2
        bottom_numpy = 3

        itemCount = len(numpyarray)
        for ind1 in range(itemCount):
            item1 = numpyarray[ind1]
            item1token = tokenIds[ind1]
            for ind2 in range(ind1 + 1, itemCount):
                item2token = tokenIds[ind2]
                item2 = numpyarray[ind2]

                # if on the same horizontal line
                if item1[top_numpy] < item2[bottom_numpy] and item2[top_numpy] < item1[bottom_numpy]:
                    # if item2 is left of item1
                    if item1[left_numpy] > item2[left_numpy] and item2token > -1:
                        resultNPArray[ind1][item2token] = 1
                    # if item1 is left of item2
                    elif item1token > -1:
                        resultNPArray[ind2][item1token] = 1

                # if left side or right side aligns (up to a threshold)
                if abs(item1[left_numpy] - item2[left_numpy]) <= threshold or abs(
                                item1[right_numpy] - item2[right_numpy]) <= threshold:
                    # if item1 above item2
                    if item1[top_numpy] < item2[top_numpy] and item2token > -1:
                        resultNPArray[ind1][item2token + self.getWordCount()] = 1
                    elif item1token > -1:
                        resultNPArray[ind2][item1token + self.getWordCount()] = 1

        if filename != None:
            np.savetxt(filename, delimiter=" ", fmt="%d", X=resultNPArray)
        return sparse.csr_matrix(resultNPArray)
