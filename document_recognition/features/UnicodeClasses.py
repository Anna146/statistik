import unicodedata
import numpy as np
from scipy import sparse

class UnicodeClassFeatures():
    CLASSES = np.array(['Cc', 'Cf', 'Co', 'Cs', 'LC', 'Ll', 'Lm', 'Lo', 'Lt', 'Lu', 'Mc', 'Me', 'Mn', 'Nd', 'Nl', 'No', 'Pc', 'Pd', 'Pe', 'Pf', 'Pi', 'Po', 'Ps', 'Sc', 'Sk', 'Sm', 'So', 'Zl', 'Zp', 'Zs', 'Fail'])
    
    def getFeatureCount(self):
        return len(self.CLASSES)

    def calcClasses(self, word):
        result = np.zeros(self.getFeatureCount())
        for x in word:
            result += (self.CLASSES == unicodedata.category(x))
        result = result / len(word)
        result[len(result)-1] = 1-np.sum(result)
        return result
        

    def extractFeatures(self, doc):
        return sparse.csr_matrix(np.array([self.calcClasses(t) for t in doc.text]))
