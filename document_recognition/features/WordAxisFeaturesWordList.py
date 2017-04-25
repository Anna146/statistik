import numpy as np
from pandas import DataFrame as df
from document_recognition.features.WordAxisFeaturesABC import WordAxisFeaturesABC
import psycopg2


class WordAxisFeaturesWordList(WordAxisFeaturesABC):

    def __init__(self, wordcount):
        super().__init__()
        self.wordcount = wordcount

    def getWordCount(self):
        return self.wordcount

    # id aus wortliste oder document
    # integer id oder -1 wenn unwichtig
    def getTokenIds(self, page: df):
        # schau mit levenshtein was der beste match für jedes Wort aus der Page ist (in der wordlist
        # return self.wordIds.get(word, -1)
        try:
            conn = psycopg2.connect(dbname='aurebu',
                                    user='aurebu',
                                    host='hub.ivi.fraunhofer.de',
                                    password='hXNtG0kCQcAEYDsCOfCn')
            cur = conn.cursor()
            cur.execute("PREPARE lev (text) AS SELECT closest_match($1)")
            result = np.zeros(len(page.text), dtype=np.int32)
            for i, word in enumerate(page.text.values):
                result[i] = self.get_best_match(word, cur)
            conn.close()
            return result
        except:
            raise

    def get_best_match(self, word, cur):
        cur.execute("EXECUTE lev(%s)", (word,))
        result = cur.fetchall()[0][0]
        return result
