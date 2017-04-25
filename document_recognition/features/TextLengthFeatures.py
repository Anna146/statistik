import numpy as np

class TextLengthFeatures():
    def getFeatureCount(self):
        return 1;

    def extractFeatures(self, doc):
        return np.array([len(t) for t in doc.text]).reshape(-1, 1)