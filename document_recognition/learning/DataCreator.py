import pickle
import numpy as np
import os
from scipy import sparse


class DataBuilder:

    def __init__(self, params_path, filter_thresh):
        params = pickle.load(open(params_path, 'rb'))
        self.features = params['features']
        self.f_keys = sorted(self.features)

        self.labels = params['labels']
        self.labels = self.labels[self.labels != 'CONFLICT']
        self.label_pos_none = np.where(self.labels == 'NONE')[0][0]
        self.n_labels = len(self.labels)

        self.n_features = 0
        for k in self.f_keys:
            v = self.features[k]
            v['mask'] = np.logical_and((v['stds'] != 0), (v['sums'] > filter_thresh))
            self.n_features += np.sum(v['mask'])

        print('# valid features :: {}'.format(self.n_features))
        self.means = np.concatenate([self.features[k]['means'][self.features[k]['mask']] for k in self.f_keys])
        self.stds = np.concatenate([self.features[k]['stds'][self.features[k]['mask']] for k in self.f_keys])

    def print_feature_list(self):
        for i, k in enumerate(sorted(self.features)):
            print('{} :: {}'.format(i, k))

    def build_for_dir(self, v_dir):
        l_file = '{}.pkl'.format(v_dir.replace('values', 'labels'))

        labels = pickle.load(open(l_file, 'rb')).values
        row_mask = labels != 'CONFLICT'
        values = self.build_value_frame(PickleValueReader(v_dir).read, row_mask)
        return values, self.get_label_matrix(labels[row_mask])

    def build_for_v_dict(self, v_dict):
        n_rows = v_dict[list(v_dict.keys())[0]].shape[0]
        return self.build_value_frame(DictValueReader(v_dict).read, np.ones(n_rows, dtype=bool))

    def build_value_frame(self, read_func, row_mask):
        values = np.empty((np.sum(row_mask), self.n_features))
        c = 0
        for k in self.f_keys:
            n_cols = np.sum(self.features[k]['mask'])
            values[:, c:c + n_cols] = np.compress(self.features[k]['mask'], read_func(k)[row_mask], axis=1)
            c += n_cols
        values -= self.means
        values /= self.stds
        return values

    def get_label_matrix(self, to_find):
        n_res = len(to_find)
        res = np.zeros((n_res, self.n_labels))
        pos = [np.where(self.labels == i)[0][0] for i in to_find]
        for i in range(n_res):
            res[i, pos[i]] = 1
        return res


class PickleValueReader:

    def __init__(self, v_dir):
        self.v_dir = v_dir

    def read(self, feature):
        return sparse_to_array(pickle.load(open(os.path.join(self.v_dir, feature), 'rb')))


class DictValueReader:
    def __init__(self, values):
        self.values = values

    def read(self, feature):
        return sparse_to_array(self.values[feature])


def sparse_to_array(data):
    if isinstance(data, sparse.csr_matrix):
        return data.toarray()
    return data