import numpy as np


class JustificationFeatures():

    def getFeatureCount(self):
        return 2;

    def extractFeatures(self, document):
        threshold = 5
        lefts = np.array([np.sum(np.abs(document.left - le) < threshold) - 1 for le in document.left])
        rights = np.array([np.sum(np.abs(document.right - r) < threshold) - 1 for r in document.right])
        return np.transpose([lefts, rights])