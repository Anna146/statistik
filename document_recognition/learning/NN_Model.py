import argparse
import tensorflow as tf
import numpy as np
import datetime
import os
import time
import pickle


class NeuralNetwork:
    """
    Base Neural Network class

    param: args ... contains all Information about layer construction and rates (learning, regularization)
    eg.: 1000 1000 1000 -> 3 Hidden Layer
    param: n_features ... size of the input layer
    param: n_labels ... size of the output layer
    """

    def __init__(self, args, n_features, n_labels):
        print('-' * 50)
        print('Tensorflow - Learning and Testing Neuronal Network')
        print(args)
        print('-' * 50)
        self.layer = args.layer
        self.reg_rate = args.reg_rate
        self.learn_rate = args.learning_rate
        self.n_epochs = args.epochs
        self.session = None
        self.n_features = n_features
        self.n_labels = n_labels

    def _get_dir_prefix(self):
        raise Exception('not yet implemented')

    def get_dir_name(self):
        return '{}_L_{}_{}-{}_e{}'.format(self._get_dir_prefix(), '-'.join(np.array(self.layer, dtype=str)),
                                                    self.learn_rate, self.reg_rate, self.n_epochs)

    def _setup_model(self):
        # Input Layer
        self.x = tf.placeholder(tf.float32, shape=[None, self.n_features], name='values')
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.n_labels])

        W_hidden = []
        b_hidden = []
        y_hidden = []

        for i, n_neurons in enumerate(self.layer):
            b_hidden.append(tf.Variable(tf.zeros([n_neurons])))
            if i == 0:
                W_hidden.append(
                    tf.Variable(tf.truncated_normal([self.n_features, n_neurons], stddev=1. / np.sqrt(self.n_features))))
                y_hidden.append(tf.nn.relu(tf.matmul(self.x, W_hidden[i]) + b_hidden[i]))
            else:
                W_hidden.append(tf.Variable(
                    tf.truncated_normal([self.layer[i - 1], n_neurons], stddev=1. / np.sqrt(self.n_features))))
                y_hidden.append(tf.nn.relu(tf.matmul(y_hidden[i - 1], W_hidden[i]) + b_hidden[i]))
        # Output Layer
        w_out = tf.Variable(tf.truncated_normal([self.layer[-1], self.n_labels], stddev=1. / np.sqrt(self.n_features)))
        b_out = tf.Variable(tf.zeros([self.n_labels]))

        regularizer = tf.nn.l2_loss(w_out)
        for w in W_hidden:
            regularizer += tf.nn.l2_loss(w)

        self.y = tf.add(tf.matmul(y_hidden[-1], w_out), b_out, name='output')
        loss_a = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y, labels=self.y_))
        loss = loss_a + (self.reg_rate * regularizer)

        # Evaluation
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(loss)

    def _init_model(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train(self, test_files, dh, eval_path):
        raise Exception('not implemented')

    def save_model(self, eval_path, global_step):
        saver = tf.train.Saver()
        saver.save(self.session, os.path.join(eval_path, 'model.tfm'), global_step=global_step)

    @staticmethod
    def get_args_parser():
        parser = argparse.ArgumentParser(description='Learning With Neural Network')
        parser.add_argument('-l', '--layer', type=int, nargs='+')
        parser.add_argument('-lr', '--learning_rate', type=float, help='learn rate between 0 and 1')
        parser.add_argument('-rr', '--reg_rate', type=float, help='regularization rate between 0 and 1')
        parser.add_argument('-e', '--epochs', type=int, help='# epochs')
        return parser


class NeuralNetworkWithNone(NeuralNetwork):

    def __init__(self, args, n_features, n_labels, pos):
        super(NeuralNetworkWithNone, self).__init__(args, n_features, n_labels)
        self._setup_model()
        weight = np.ones(self.n_labels)
        self.weight_pos = pos
        if self.weight_pos is not None:
            weight[self.weight_pos] = 0.001
        ## penalizing and loss a
        # self.y = tf.multiply(tf.nn.softmax(self.y), weight, name='output')
        # loss_a = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y+(1e-10)), reduction_indices=[1]))
        ## penalizing and loss b
        self.y = tf.multiply(self.y, weight, name='output')

    def _get_dir_prefix(self):
        return 'NN_WithNone_WP_{}'.format(self.weight_pos)

    def train(self, train_files, dh, eval_path):
        start_time = datetime.datetime.now()
        accs = []
        with self.session.as_default():
            try:
                for epoch in range(self.n_epochs):
                    for i in range(len(train_files)):
                        values, labels = dh.build_for_dir(train_files[i])
                        current_acc = self.session.run(self.accuracy, feed_dict={self.x: values, self.y_: labels})
                        current_acc = 0 if np.isnan(current_acc) else current_acc
                        accs.append(current_acc)
                        self.train_step.run(feed_dict={self.x: values, self.y_: labels})
                        print('\rEpoch [{}/{}] File [{}/{}] Acc : {}'.format(epoch+1, self.n_epochs, i+1, len(train_files),
                                                                           current_acc), end='')
                    self.save_model(eval_path, epoch)
            except KeyboardInterrupt:
                print('\t --> abborted')
        print()
        print('Trained in :: {}'.format(datetime.datetime.now() - start_time))
        return accs


class NeuralNetworkRandomPick(NeuralNetwork):

    def __init__(self, args, n_features, n_labels, file_batch, pick_count):
        super(NeuralNetworkRandomPick, self).__init__(args, n_features, n_labels)
        self._setup_model()
        self._init_model()
        self.file_batch = file_batch
        self.pick_count = pick_count

    def _get_dir_prefix(self):
        return 'NN_RNP_FB_{}_PC_{}'.format(self.file_batch, self.pick_count)

    def train(self, train_files, dh, eval_path):
        start_time = datetime.datetime.now()
        accs = []
        times = []
        with self.session.as_default():
            try:
                for epoch in range(self.n_epochs):
                    for i in range(int(len(train_files)/self.file_batch)):
                        t1 = time.time()
                        values, labels = self.pick(
                            train_files[int(i*self.file_batch):int(i*self.file_batch)+self.file_batch], dh)
                        pickTime = np.round(time.time() - t1, decimals=3)
                        t1 = time.time()
                        current_acc = self.session.run(self.accuracy, feed_dict={self.x: values, self.y_: labels})
                        testTime = np.round(time.time() - t1, decimals=3)
                        t1 = time.time()
                        accs.append(0 if np.isnan(current_acc) else current_acc)
                        self.train_step.run(feed_dict={self.x: values, self.y_: labels})
                        trainTime = np.round(time.time() - t1, decimals=3)
                        times.append([pickTime, testTime, trainTime])
                        print('\rEpoch [{}/{}] File [{}/{}] Acc : {} \t [{}, {}, {}]'.format(epoch+1, self.n_epochs, i+1,
                                                                             int(len(train_files)/self.file_batch),
                                                                             accs[-1], pickTime, testTime, trainTime), end='')
                    if (epoch % 5) == 0:
                        self.save_model(eval_path, epoch)
                self.save_model(eval_path, self.n_epochs)
            except KeyboardInterrupt:
                print('\t --> abborted')
                self.save_model(eval_path, self.n_epochs)
        print()
        print('Trained in :: {}'.format(datetime.datetime.now() - start_time))
        pickle.dump(times, open(os.path.join(eval_path, 'times.pkl'), 'wb'))
        pickle.dump(accs, open(os.path.join(eval_path, 'accs_train.pkl'), 'wb'))
        return accs

    def pick(self, selected_files, dh):
        res = []
        buffer_valid_data = []
        buffer_valid_label = []
        buffer_valid_label_id = []
        for f in selected_files:
            values, labels = dh.build_for_dir(f)
            l_tmp = np.argmax(labels, 1)
            for label_id in np.unique(l_tmp):
                choice = np.random.choice(np.where(l_tmp == label_id)[0], self.pick_count, True)
                if label_id == dh.label_pos_none:
                    res.append((values[choice], labels[choice]))
                else:
                    buffer_valid_data.append(values[choice])
                    buffer_valid_label.append(labels[choice])
                    buffer_valid_label_id.append(label_id)

        buffer_valid_data = np.array(buffer_valid_data)
        buffer_valid_label = np.array(buffer_valid_label)
        buffer_valid_label_id = np.array(buffer_valid_label_id)

        for l in np.unique(buffer_valid_label_id):
            mask = buffer_valid_label_id == l
            valid_data = np.concatenate(buffer_valid_data[mask])
            valid_labels = np.concatenate(buffer_valid_label[mask])
            v = valid_data[np.random.choice(np.arange(len(valid_data)), self.pick_count * self.file_batch, True)]
            l = valid_labels[np.random.choice(np.arange(len(valid_labels)), self.pick_count * self.file_batch, True)]
            res.append((v, l))

        np.random.shuffle(res)
        res = np.array(res)
        v = np.concatenate(res[:, 0])
        l = np.concatenate(res[:, 1])
        return v, l
