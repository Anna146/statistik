import datetime
import os
import pickle

import numpy as np

import document_recognition.learning.NN_Model as NN_Models
from document_recognition.learning.DataCreator import DataBuilder
from document_recognition.learning.ModelTester import ModelTester


def get_shuffled_train_test(path, train_rate, seed=0):
    files = np.array(list(os.path.join(path, f) for f in sorted(os.listdir(path))))
    rand = np.arange(len(files))
    np.random.seed(seed)
    np.random.shuffle(rand)
    train = files[rand[:int(len(files) * train_rate)]]
    test = files[rand[-int(len(files) * (1 - train_rate)):]]
    return train, test


if __name__ == '__main__':
    print('\n\n')
    start = datetime.datetime.now()

    parser = NN_Models.NeuralNetwork.get_args_parser()
    parser.add_argument('path', type=str, help='path to the feature dir including labels, values and params')
    parser.add_argument('-ts', '--train_size', type=float, help='train size between 0 and 1')
    parser.add_argument('-ff', '--feature_filter', type=int, default=0, help='feature filter size -> sums > ff')
    args = parser.parse_args()

    dh = DataBuilder(os.path.join(args.path, 'params.pkl'), filter_thresh=args.feature_filter)

    model = NN_Models.NeuralNetworkRandomPick(args, dh.n_features, dh.n_labels, file_batch=20, pick_count=10)

    date = start.strftime('%y-%m-%d_%H-%M')
    dirname = '{}_{}_TS_{}_FF_{}'.format(date, model.get_dir_name(), int(args.train_size * 100), args.feature_filter)
    eval_path = os.path.join(args.path, 'results', dirname)
    os.makedirs(eval_path, exist_ok=True)
    pickle.dump(args, open(os.path.join(eval_path, 'args.pkl'), 'wb'))
    pickle.dump(dh, open(os.path.join(eval_path, 'databuilder.pkl'), 'wb'))

    feature_keys_train, feature_keys_test = get_shuffled_train_test(os.path.join(args.path, 'values'), args.train_size)
    pickle.dump(feature_keys_test, open(os.path.join(eval_path, 'test_files.pkl'), 'wb'))
    print('>> Start Training <<')
    model.train(feature_keys_train, dh, eval_path)

    print('>> Testing <<')
    start_time = datetime.datetime.now()
    mt = ModelTester(eval_path)
    mt.test_for_files(feature_keys_test)
    print('Finished in {}'.format(datetime.datetime.now() - start))

