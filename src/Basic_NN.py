from random import randint

import numpy
import tensorflow as tf
from aptsources.sourceslist import NullMatcher
from scipy import ndimage
from tensorflow.examples.tutorials.mnist import mnist, input_data
from tensorflow.contrib.tensorboard.plugins import projector
import os


# Processing
PATH = os.getcwd()
STEPS = 1000

LOGDIR = PATH + '/log/basic_weight_visualization'
NUM_OUTPUTS = 3

# Methods

from src.Board import Board
from src.Logger import Logger
from src.Log import Log

import pickle
import numpy as np
import collections

LOCATION = os.getcwd()
FILE_NAME_LOGGER = LOCATION + '/data/log/Padding_Test.pkl'
FILE_PATH_SESSION = LOCATION + '/data/session/'
MODEL_NAME = 'Basic_NN'

NUM_EPOCHS = 10
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.99
KEEP_PROP = 1


class Basic_NN:

    def __init__(self):
        # Load logger data for training
        with open(FILE_NAME_LOGGER, 'rb') as f:
            self.logger = pickle.load(f)

        # Neural Network Architecture
        # Input and Labels
        self.x_input = tf.placeholder(tf.float32, shape=[None, Board.NUM_ROWS * Board.NUM_COLUMNS * len(Board.EMPTY_CELL)])
        self.y_label = tf.placeholder(tf.float32, shape=[None, Board.NUM_COLUMNS])
        self.Rewards = tf.placeholder("float", shape=[None])  # Reward input to weight input
        self.keep_prob = tf.placeholder(tf.float32)

        # Neural Network layer architecture
        layer_1 = self.full_layer(self.x_input, 1024, name='layer_1')
        layer_2 = self.full_layer(layer_1, 1024, name='layer_2')
        layer_3 = self.full_layer(layer_2, Board.NUM_COLUMNS, name='layer_3')
        self.y_output = layer_3

        # Operations on the NN
        self.cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_output, labels=self.y_label) * self.Rewards)
        self.loss_operation = self.cross_entropy
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=LEARNING_RATE_DECAY)
        self.train_step = self.optimizer.minimize(self.loss_operation)
        self.correct_prediction = tf.equal(tf.argmax(self.y_output, 1), tf.argmax(self.y_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.global_step = 0
        self.prediction_rnn = self.y_output

        # Tensorboard Scalars to be saved for tensorboard
        tf.summary.scalar("cross_entropy", self.cross_entropy)
        tf.summary.scalar("accuracy", self.accuracy)
        self.writer = tf.summary.FileWriter(LOGDIR)

        # Operation with merges all summries for tensorboard
        self.summ = tf.summary.merge_all()

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

        # Saver to save model
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        # Run the initializer
        self.sess.run(self.init)


    # Defines weight varibale according to a given shape
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


    # Defines bias variable according to a given shape
    def bias_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # Creates a fully connected layer
    def full_layer(self, input_data, out_size, name):
        with tf.name_scope(name):
            in_size = int(input_data.get_shape()[1])
            w = self.weight_variable([in_size, out_size])
            b = self.bias_variable([out_size])
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
        return tf.nn.softmax(tf.matmul(input_data, w) + b)

    # Save the the actual session
    def save(self):
        self.saver.save(self.sess, FILE_PATH_SESSION+MODEL_NAME, global_step=1000)

    # Save the the actual session
    def save(self, global_step):
        self.saver.save(self.sess, FILE_PATH_SESSION+MODEL_NAME, global_step=global_step)

    # Load the last saved session
    def load(self, session_name):
        tf.reset_default_graph()
        filename = session_name + '.meta'
        self.saver = tf.train.import_meta_graph(FILE_PATH_SESSION+filename)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(FILE_PATH_SESSION))

    # Predict next "winning" game state on the board matrix using RNN
    def predict(self, x_input):
        x = np.array(x_input).reshape((1, Board.NUM_COLUMNS * Board.NUM_ROWS * len(Board.EMPTY_CELL)))
        predicted = self. sess.run([self.prediction_rnn], feed_dict={self.x_input: x})
        predicted = np.array(predicted).reshape(Board.NUM_COLUMNS)
        return predicted

    # Create a weight array that gives discount to every step
    def discount_rewards(self, rewards, gamma):
        counter_rewards = collections.Counter(rewards)
        num_one = counter_rewards[1]
        num_minus_one = counter_rewards[-1]
        exponent = num_one + num_minus_one
        for i in range(0, num_one+num_minus_one):
            rewards[i] = rewards[i] * pow(gamma, exponent)
            exponent -= 1

    # trains NN on a single batch (log)
    def _train(self, log):
        batch_x = log.get_all_states(padding=True, including_last=False)
        batch_y = log.get_all_actions(cell_type=None, padding=True)
        rewards = log.get_rewards()
        self.discount_rewards(rewards=rewards, gamma=0.99)
        batch_x = np.array(batch_x).reshape((42, Board.NUM_COLUMNS * Board.NUM_ROWS * len(Board.EMPTY_CELL)))
        batch_y = np.array(batch_y).reshape((42, Board.NUM_COLUMNS))
        reward = np.array(rewards).reshape((42))

        self.global_step += 1
        # Show progress
        if self.global_step % 100 == 0:
            cross_entropy = self.sess.run([self.cross_entropy],
                                     feed_dict={self.x_input: batch_x, self.y_label: batch_y, self.Rewards: reward,
                                                self.keep_prob: 1.0})
            print("step {}, cross_entropy {}".format(self.global_step, cross_entropy))

        # Run a trainstep and summarize tensorboard stuff
        [acc, s] = self.sess.run([self.train_step, self.summ],
                            feed_dict={self.x_input: batch_x, self.y_label: batch_y, self.Rewards: reward,
                                       self.keep_prob: 0.8})
        self.writer.add_summary(s, self.global_step)
        self.writer.flush()

    def train(self, log):
            self.sess.run(self.init)
            self.writer.add_graph(self.sess.graph)
            for epoch in range(0, NUM_EPOCHS):
                if log is None:
                    # Train the model
                    for log in self.logger.logs:
                        self._train(log)
                else:
                    self._train(log)



#basic_nn = Basic_NN()
#basic_nn.train(log=None)
#basic_nn.save(126)