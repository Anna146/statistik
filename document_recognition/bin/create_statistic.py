import argparse
import numpy as np
import pandas
import pylab as plt
import os
from collections import Counter

parser = argparse.ArgumentParser(description='Create Stats For Labeled Data')
parser.add_argument('path', type=str, help='the path for the ')
parser.add_argument('--out', type=str, help='output filename', default='./labels_in_files.png')


def create_stat(path, out_file):
    c = Counter()
    file_list = list(filter(lambda x: x.endswith('.csv'), os.listdir(path)))
    for f in file_list:
        ue = pandas.read_csv(os.path.join(path, f), sep=' ').label.unique()
        c.update(ue)

    plt.figure(figsize=(15, 5))
    labels = np.array(list(c.keys()))
    keys_sorted = np.array(sorted(c.keys()))
    keys_idx = [np.where(labels == k)[0][0] for k in keys_sorted]

    plt.bar(np.arange(len(c.keys())), np.array(list(c.values()))[keys_idx])
    plt.xticks(np.arange(len(c.keys())), keys_sorted, rotation=90)
    plt.tight_layout()
    plt.savefig(out_file)

if __name__ == '__main__':
    args = parser.parse_args()
    create_stat(args.path, args.out)
