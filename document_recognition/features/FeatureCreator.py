from document_recognition.preprocess.PageAlign import alignPages
from document_recognition.features.WordAxisFeaturesFromDocument import WordAxisFeaturesFromDocument
from document_recognition.features.WordAxisFeaturesWordList import WordAxisFeaturesWordList

import os
import pandas
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import pickle
from scipy import sparse

CSV_CONTENT_DTYPES = {0: np.int16, 1: np.int16, 2: np.int16, 3: np.int16, 4: np.int16, 7: object}


class FeatureCreator:

    def __init__(self, feature_setup):
        self.feature_setup = feature_setup

    def dump(self, input, output):
        input_files = [os.path.join(input, f) for f in filter(lambda x: x.endswith('.json.csv'), os.listdir(input))]
        self.value_path = os.path.join(output, 'values')
        self.label_path = os.path.join(output, 'labels')
        os.makedirs(self.value_path, exist_ok=True)
        os.makedirs(self.label_path, exist_ok=True)
        n_cores = mp.cpu_count()
        print('Calculation on {} cores'.format(n_cores))
        pool = mp.Pool(processes=n_cores)
        pool.map(self.dump_for_file, input_files)
        pool.close()
        pool.join()

    def dump_for_file(self, file):
        filename = os.path.basename(file)
        pd = read_file(file)
        if pd is not None:
            feature_base_out = os.path.join(self.value_path, filename.split('.')[0])
            os.makedirs(feature_base_out, exist_ok=True)
            work = [(pd, f, os.path.join(feature_base_out, type(f).__name__)) for f in self.feature_setup]
            with ThreadPoolExecutor() as e:
                list(e.map(extract_feature, work))
            pickle.dump(pd['label'], open(os.path.join(self.label_path, filename.replace('json.csv', 'pkl')), 'wb'))


def create_for_test(data, feature_setup):
    values = {}
    for feature in feature_setup:
        if isinstance(feature, WordAxisFeaturesFromDocument):
            print("<debug> WordAxisFeaturesFromDocument found. Replace by ...WordList")
            values[type(feature).__name__] = \
                WordAxisFeaturesWordList(feature.getWordCount()).extractFeatures(data)
        else:
            values[type(feature).__name__] = feature.extractFeatures(data)
    return values


def read_file(file):
    if is_file_valid(file):
        pd = pandas.read_csv(file, sep=' ', encoding='utf8', na_filter=False, dtype=CSV_CONTENT_DTYPES)
        if len(pd.left) == 0:
            print('{} has just header!!!'.format(file))
            return
        return alignPages(pd)
    return


def is_file_valid(file):
    return os.path.getsize(file) > 0 and file.endswith('.csv')


def extract_feature(work):
    data, feature_extractor, path = work
    values = feature_extractor.extractFeatures(data)
    pickle.dump(values, open(path, 'wb'))