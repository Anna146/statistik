import os
import numpy as np
import pickle
import sklearn.metrics as skmetr
import matplotlib as mpl
mpl.use('Agg')
import pylab as plt
import argparse

parser = argparse.ArgumentParser(description='Create Evaluation Plots')
parser.add_argument('model_directories', type=str, nargs='+', help='directories')


class Evaluator(object):

    def __init__(self, eval_path, labels, real, predicted):
        self.labels = labels
        self.eval_path = eval_path
        self.real = real
        self.predicted = predicted
        self.total_real = np.concatenate(real)
        self.total_pred = np.concatenate(predicted)

    def create(self):
        self.create_accuracy()
        self.create_confusion()
        self.create_score(skmetr.recall_score, 'recall')
        self.create_score(skmetr.f1_score, 'f1')
        self.create_score(skmetr.precision_score, 'precisicn')

    def create_accuracy(self):
        accs = []
        for i in range(len(self.real)):
            accs.append(sum(self.real[i] == self.predicted[i]) / len(self.real[i]))

        plt.figure(figsize=(10, 5))
        plt.plot(accs)
        plt.ylim([0, 1])
        plt.title('Accuracy')
        plt.savefig(os.path.join(self.eval_path, 'accuracy.png'))
        plt.tight_layout()

    def create_score(self, score_func, name):
        values = score_func(self.total_real, self.total_pred, average=None)
        plt.figure(figsize=(15, 5))
        plt.bar(range(len(values)), values)
        plt.ylim([0, 1])
        plt.xticks(range(len(values)), self.labels, rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_path, '{}_{}'.format(name, np.mean(values)).replace('.', '-')))

    def create_confusion(self):
        cc = skmetr.confusion_matrix(self.total_real, self.total_pred)
        cc = cc / np.sum(cc, 0)
        plt.figure(figsize=(15, 10))

        plt.imshow(cc, cmap=plt.cm.Blues, vmin=0, vmax=1)
        plt.colorbar()
        plt.xticks(np.arange(len(db.labels)), db.labels, rotation=90)
        plt.yticks(np.arange(len(db.labels)), db.labels)  # [::-1])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.eval_path, 'confusion.png'))


def create_time_plot(eval_path, times):
    plt.figure(figsize=(15, 10))
    plt.plot(times[:, 0], 'r', label='pick')
    plt.plot(times[:, 1], 'b', label='test')
    plt.plot(times[:, 2], 'g', label='train')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(eval_path, 'times.png'))


if __name__ == '__main__':
    args = parser.parse_args()
    for dir in args.model_directories:
        print(dir)
        db = pickle.load(open(os.path.join(dir, 'databuilder.pkl'), 'rb'))
        real = pickle.load(open(os.path.join(dir, 'real.pkl'), 'rb'))
        pred = pickle.load(open(os.path.join(dir, 'predicted.pkl'), 'rb'))
        times = np.array(pickle.load(open(os.path.join(dir, 'times.pkl'), 'rb')))
        Evaluator(dir, db.labels, real, pred).create()

        create_time_plot(dir, times)