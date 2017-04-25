import numpy as np
import pickle
from scipy import sparse
import os
import argparse
import pandas
import copy
import multiprocessing as mp

parser = argparse.ArgumentParser(description='Filter Data by STD (STD == 0)')
parser.add_argument('path', nargs=1)
parser.add_argument('ff', type=int, nargs=1)

def reduce(stuff):
    path, doc, features, masks = stuff
    os.mkdir(os.path.join(path, 'values_new', doc))
    for feature in features:
        p = os.path.join(path, 'values_new', doc, feature)
        data = pickle.load(open(os.path.join(path, 'values', doc, feature), 'rb'))
        if type(data) == sparse.csr_matrix:
            data = data.toarray()
            new_data = data[:, masks[feature]]
            new_data = sparse.csr_matrix(new_data)
            pickle.dump(new_data, open(p, 'wb'))
        elif type(data) == pandas.DataFrame:
            data = data.values
            new_data = data[:, masks[feature]]
            pickle.dump(new_data, open(p, 'wb'))
        elif type(data) == np.ndarray:
            new_data = data[:, masks[feature]]
            pickle.dump(new_data, open(p, 'wb'))

if __name__ == '__main__':
    print('--- Start Filter ---')
    args = parser.parse_args()
    path = args.path[0]
    print(path)
    params = pickle.load(open(os.path.join(path, 'params.pkl'), 'rb'))
    new_params = copy.deepcopy(params)

    masks = {}
    for feature in params['features']:
        m1 = params['features'][feature]['stds'] != 0
        m2 = params['features'][feature]['sums'] > args.ff
        masks[feature] = m1*m2
        new_params['features'][feature]['stds'] = new_params['features'][feature]['stds'][masks[feature]]
        new_params['features'][feature]['sums'] = new_params['features'][feature]['sums'][masks[feature]]

    for feature in params['features']:
        print(feature, end=' ')
        print(len(params['features'][feature]['stds']), end='/')
        print(len(new_params['features'][feature]['stds']), end=' \t')
        print(len(params['features'][feature]['sums']), end='/')
        print(len(new_params['features'][feature]['sums']), end=' ')
        print()
    pickle.dump(new_params, open(os.path.join(path, 'params_new.pkl'), 'wb'))
    print()
    os.mkdir(os.path.join(path, 'values_new'))

    features = list(params['features'].keys())
    print(features)
    stuff = [(path, l, features, masks) for l in os.listdir(os.path.join(path, 'values'))]
    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(reduce, stuff)