class PositionFeatures():
    def getFeatureCount(self):
        return 5;

    def extractFeatures(self, document):
        return document.ix[:, :5].values