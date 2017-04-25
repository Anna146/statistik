import pickle
import numpy as np
import multiprocessing as mp
import os
from scipy import sparse

N_CPU = mp.cpu_count()


class ParameterExtractor:

    def __init__(self, labelpath, valuepath):
        self.label_path = labelpath
        self.value_path = valuepath

    def generate(self):
        res = {}
        res['labels'] = self.get_unique_labels()
        res['features'] = self.get_standardization_params()
        return res

    def get_unique_labels(self):
        label_files = [os.path.join(self.label_path, f) for f in os.listdir(self.label_path)]

        pool = mp.Pool(processes=N_CPU)
        res = np.unique(np.concatenate(pool.map(extract_labels, label_files)))
        pool.close()
        pool.join()
        return res

    def get_standardization_params(self):
        value_files = []
        for dirname, dirnames, filenames in os.walk(self.value_path):
            for subdirname in dirnames:
                subpath = os.path.join(dirname, subdirname)
                value_files.append([os.path.join(subpath, f) for f in os.listdir(subpath)])

        pool = mp.Pool(processes=N_CPU)
        factor = (1 / len(value_files))
        means, stds, sums = extract_standardization_params(value_files[0])
        means *= factor
        stds *= factor

        for r in pool.imap(extract_standardization_params, value_files[1:]):
            means += factor * r[0]
            stds += factor * r[1]
            sums += r[2]
        pool.close()
        pool.join()

        masks = extract_feature_indizes(value_files[0])
        res = {}
        for k in masks:
            v = masks[k]
            res[k] = {'means': means[v], 'stds': stds[v], 'sums': sums[v]}

        return res


def extract_feature_indizes(value_files):
    tmp = [(os.path.basename(f), pickle.load(open(f, 'rb')).shape[1]) for f in value_files]
    total = np.sum(e[1] for e in tmp)

    masks = {}
    c = 0
    for i in range(len(tmp)):
        values = np.zeros(total, dtype=bool)
        values[c:c+tmp[i][1]] = True
        masks[tmp[i][0]] = values
        c += tmp[i][1]

    return masks


def extract_labels(f):
    return np.unique(pickle.load(open(f, 'rb')))


def extract_standardization_params(value_files):
    mean = []
    std = []
    sum = []
    for f in value_files:
        data = pickle.load(open(f, 'rb'))
        if type(data) == sparse.csr_matrix:
            data = data.toarray()
        mean.append(np.mean(data, axis=0))
        std.append(np.std(data, axis=0))
        sum.append(np.sum(data, axis=0))

    return np.concatenate(mean), np.concatenate(std), np.concatenate(sum)
