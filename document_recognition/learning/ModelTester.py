from document_recognition.util.FileListSorter import sort_nicely
import pickle
import os
import tensorflow as tf
from document_recognition.features.FeatureCreator import create_for_test
import numpy as np
from datetime import datetime


class ModelTester:

    def __init__(self, path_to_model):
        self.path_to_model = path_to_model
        self.databuilder = pickle.load(open(os.path.join(path_to_model, 'databuilder.pkl'), 'rb'))
        fg_path = os.path.join((os.path.dirname(os.path.dirname(path_to_model))), 'feature_setup.pkl')
        self.feature_setup = pickle.load(open(fg_path, 'rb'))

        tf.reset_default_graph()
        self.g = tf.Graph().as_default()
        self.session = tf.Session()
        metas = list(filter(lambda x: 'meta' in x, os.listdir(path_to_model)))
        sort_nicely(metas)
        print(metas[-1])
        print('Load Model ', end='')
        s = datetime.now()
        saver = tf.train.import_meta_graph(os.path.join(path_to_model, metas[-1]))
        saver.restore(self.session, os.path.join(path_to_model, metas[-1].split('.meta')[0]))
        self.values = tf.get_default_graph().get_tensor_by_name('values:0')
        self.output = tf.get_default_graph().get_tensor_by_name('output:0')
        print('in {}'.format(datetime.now() - s))

    def apply_json(self, data):
        s = datetime.now()
        print('Create Data ', end='')
        features = create_for_test(data, self.feature_setup)
        features = self.databuilder.build_for_v_dict(features)
        print('in {}'.format(datetime.now() - s))
        print('Evaluate ', end='')
        s = datetime.now()
        print('in {}'.format(datetime.now() - s))
        res = self.apply(features)
        return res, self.databuilder.labels[np.argmax(res, 1)]

    def apply_dir(self, dir):
        values, labels = self.databuilder.build_for_dir(dir)
        return self.apply(values), labels

    def apply(self, values):
        return self.session.run(tf.nn.softmax(self.output), feed_dict={self.values: values})

    def test_for_files(self, feature_keys_test):
        real = []
        predicted = []
        probabilities = []
        for i in range(len(feature_keys_test)):
            prob, label = self.apply_dir(feature_keys_test[i])
            probabilities.append(prob)
            predicted.append(np.argmax(prob, 1))
            real.append(np.argmax(label, 1))
            print('\rFile [{}/{}]'.format(i + 1, len(feature_keys_test)), end='')

        pickle.dump(real, open(os.path.join(self.path_to_model, 'real.pkl'), 'wb'))
        pickle.dump(predicted, open(os.path.join(self.path_to_model, 'predicted.pkl'), 'wb'))
        pickle.dump(predicted, open(os.path.join(self.path_to_model, 'probs.pkl'), 'wb'))

    def test(self):
        test_files = pickle.load(open(os.path.join(self.path_to_model, 'test_files.pkl'), 'rb'))
        self.test_for_files(test_files)