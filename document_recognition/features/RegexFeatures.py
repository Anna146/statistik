import regex
import numpy as np


class RegexFeatures():
    ALWAYS_FAILING = regex.compile('(*FAIL)')
    REGEX_PROTOTYPES = [
        (50, "\\d+"),  # numeric
        (15, "(?:\d{1,3})(?:\.?\d{3})+(?:,\d{2})?€?"),  # currency
        (50, "\\p{L}+")  # Text
    ]
    REGEX_CACHE = {}
    REGEX_RESULTS = {}

    def getThreshold(self, wordLen):
        return wordLen // 5

    def getRegexes(self, threshold):
        if (threshold in self.REGEX_CACHE):
            return self.REGEX_CACHE[threshold]
        else:
            x = [regex.compile('(?>' + r + '){e<=' + str(threshold) + '}') if threshold <= self.getThreshold(
                l) else self.ALWAYS_FAILING for (l, r) in self.REGEX_PROTOTYPES]
            self.REGEX_CACHE[threshold] = x
            return x

    def generateMatches(self, token):
        threshold = self.getThreshold(len(token))
        if (token in self.REGEX_RESULTS):
            return self.REGEX_RESULTS[token]
        res = np.array([0 if rex.fullmatch(token) == None else 1 for rex in self.getRegexes(threshold)])
        self.REGEX_RESULTS[token] = res
        return res

    def getFeatureCount(self):
        return len(self.REGEX_PROTOTYPES);

    def extractFeatures(self, doc):
        return np.array([self.generateMatches(t) for t in doc.text])