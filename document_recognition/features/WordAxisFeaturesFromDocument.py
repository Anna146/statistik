from pandas import DataFrame as df
from document_recognition.features.WordAxisFeaturesABC import WordAxisFeaturesABC

threshold = 2

CORE_DATA = "CORE_DATA"
# CORE DATA
WORD = "text"
LEFT = "left"
RIGHT = "right"
TOP = "top"
BOTTOM = "bottom"
PAGE = "page"

# CALCULATED DATA
H_AXIS = "H_AXIS"
V_AXIS = "V_AXIS"
RIGHT_OF = "right_of"


class WordAxisFeaturesFromDocument(WordAxisFeaturesABC):
    wordIds = dict()
    wordcount = 0

    def __init__(self, wordcount):
        super().__init__()
        self.wordcount = wordcount

    # token zahl (55000) entweder übergeben oder aus wordlist
    def getWordCount(self):
        return self.wordcount

    # id aus wortliste oder document
    # integer id oder -1 wenn unwichtig
    def getTokenIds(self, page:df):
        return page["token"].values

