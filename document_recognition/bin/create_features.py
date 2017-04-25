from document_recognition.features.TextLengthFeatures import TextLengthFeatures
from document_recognition.features.PositionFeatures import PositionFeatures
from document_recognition.features.JustificationFeatures import JustificationFeatures
from document_recognition.features.UnicodeClasses import UnicodeClassFeatures
from document_recognition.features.RegexFeatures import RegexFeatures
from document_recognition.features.WordAxisFeaturesFromDocument import WordAxisFeaturesFromDocument
from document_recognition.features.BoundingBoxFeatures import BoundingBoxFeatures
from document_recognition.features.FeatureCreator import FeatureCreator
from document_recognition.features.ParameterExtractor import ParameterExtractor

import datetime
import os
import argparse
import pickle

parser = argparse.ArgumentParser(description='Create Features')
parser.add_argument('preprocess', type=str, help='path of the pre-process out files')
parser.add_argument('out', type=str, help='path for the created features')


if __name__ == '__main__':
    args = parser.parse_args()

    print('Load Words.hist')
    word_hist_filename = os.path.join(args.preprocess, 'words.hist')

    with open(word_hist_filename, encoding='utf8') as f:
        word_count = len(f.readlines())

    FEATURE_SET = (PositionFeatures(),
                   TextLengthFeatures(),
                   JustificationFeatures(),
                   UnicodeClassFeatures(),
                   RegexFeatures(),
                   WordAxisFeaturesFromDocument(word_count),
                   BoundingBoxFeatures())

    FEATURE_SET_NAMES = [type(f).__name__ for f in FEATURE_SET]

    fc = FeatureCreator(FEATURE_SET)

    start_time = datetime.datetime.now()
    '''CREATE FEATURES'''
    print(FEATURE_SET_NAMES)
    print('Feature Creation >> ', end=' ')
    fc.dump(args.preprocess, args.out)
    feature_setup_path = os.path.join(args.out, 'feature_setup.pkl')
    if os.path.exists(feature_setup_path):
        import numpy as np
        print('old features available')
        last_features = pickle.load(open(feature_setup_path, 'rb'))
        last = []
        for f in last_features:
            if type(f).__name__ not in FEATURE_SET_NAMES:
                last.append(f)
        FEATURE_SET = np.concatenate((FEATURE_SET, last))
    pickle.dump(FEATURE_SET, open(feature_setup_path, 'wb'))
    print('finished in {}'.format(datetime.datetime.now() - start_time))

    '''EXTRACT PARAMS FOR STANDARDIZATION'''
    print('Parameter Extraction >> ', end=' ')
    start_time = datetime.datetime.now()
    params = ParameterExtractor(labelpath=fc.label_path, valuepath=fc.value_path).generate()
    pickle.dump(params, open(os.path.join(args.out, 'params.pkl'), 'wb'))

    print('finished in {}'.format(datetime.datetime.now() - start_time))
