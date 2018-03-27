from random import Random, shuffle

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from src.Board import Board
import pickle
import os
from src.Log_Analyzer import Log_Analyzer

LOCATION = os.getcwd()
FILE_NAME_LOGGER = LOCATION + '/data/log/Basic_NN Tue Mar 20 20:51:50 2018.pkl'
FILE_PATH_SESSION = LOCATION + '/data/session/'
FILE_NAME_BIASES = LOCATION + '/data/model/rnn_biases.pkl'
MODEL_NAME = 'LSTM_HIDDEN_1024_rand_vs_rand'
X_CELL_FILTER_CONV_1 = 4
Y_CELL_FILTER_CONV_1 = 4
NUM_FILTER_CONV_1_OUT = 20
X_CELL_FILTER_CONV_2 = 4
Y_CELL_FILTER_CONV_2 = 4
NUM_FILTER_CONV_2_IN = NUM_FILTER_CONV_1_OUT
NUM_FILTER_CONV_2_OUT = 20
KEEP_PROP_DROPOUT = 0.8
LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.99
NUM_HIDDEN = 1024


class AI:

    def __init__(self, network_type):
        # Load data
        with open(FILE_NAME_LOGGER, 'rb') as f:
            self.logger = pickle.load(f)

        # Build RNN Graph

        # Training Parameters
        self.learning_rate = LEARNING_RATE
        self.learning_rate_decay = LEARNING_RATE_DECAY
        self.training_steps = 2000
        self.batch_size = 1
        self.display_step = 10
        self.global_step = 0
        self.actual_game_actions = 0

        # Network Parameters
        self.num_input = Board.NUM_COLUMNS  # Board  dimensions
        self.timesteps = int((Board.NUM_COLUMNS * Board.NUM_ROWS) / 2)  # timesteps
        self.num_hidden = NUM_HIDDEN  # hidden layer num of features
        self.num_classes = Board.NUM_COLUMNS  # Board dimension (prediction)

        # tf Graph input
        # RNN
        if network_type == 'rnn':
            self.X_rnn = tf.placeholder("float", [None, self.timesteps, self.num_classes])  # Input of RNN
            self.Y_rnn = tf.placeholder("float", [None, self.timesteps, self.num_classes])  # Output of RNN
            self.Outputs = None
            # Define weights and biases
            self.weights = {
                'out': tf.Variable(tf.random_normal([21,self.num_hidden, self.num_classes], mean=0, stddev=0.01))
            }
            self.biases = {
                'out': tf.Variable(tf.random_normal([self.num_classes], mean=0, stddev=0.01))
            }
            self.logits_rnn = self.RNN()
            self.prediction_rnn = tf.nn.softmax(self.logits_rnn)

            # Define loss and optimizer
            self.Rewards = tf.placeholder("float", shape=[None])  # Reward input to weight input
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_rnn, labels=self.Y_rnn)
            self.loss_operation = tf.reduce_sum(self.Rewards * self.cross_entropy)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_operation = self.optimizer.minimize(self.loss_operation)

            # Evaluate model (with test logits, for dropout to be disabled)
            self.correct_prediction = tf.equal(tf.argmax(self.prediction_rnn, 1), tf.argmax(self.Y_rnn, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))



        # CNN
        elif network_type == 'cnn':
            self.X_cnn = tf.placeholder("float", [None, Board.NUM_ROWS, Board.NUM_COLUMNS, len(Board.EMPTY_CELL)])
            self.Y_cnn = tf.placeholder("float", [None, self.num_classes])
            self.logits_cnn = self.CNN()
            self.prediction_cnn = tf.nn.sigmoid(self.logits_cnn)
            # Define loss and optimizer
            self.Rewards = tf.placeholder("float", shape=[None])  # Reward input to weight input
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_cnn, labels=self.Y_cnn)
            self.loss_operation = tf.reduce_sum(self.Rewards * self.cross_entropy)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.learning_rate_decay)
            self.train_operation = self.optimizer.minimize(self.loss_operation)
            # Evaluate model (with test logits, for dropout to be disabled)
            self.correct_prediction = tf.equal(tf.argmax(self.prediction_cnn, 1), tf.argmax(self.Y_cnn, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

        # Saver to save model
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        # Run the initializer
        self.sess.run(self.init)

    def RNN(self):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(self.X_rnn, self.timesteps, 1)

        # Define a rnn cell with tensorflow
        #rnn_cell = rnn.BasicRNNCell(self.num_hidden)
        rnn_cell = rnn.LSTMCell(self.num_hidden, activation=tf.nn.relu, forget_bias=0.5)

        # Get RNN cell output
        self.outputs, states = rnn.static_rnn(rnn_cell, x, sequence_length=[self.timesteps], dtype=tf.float32)


        #full_1_drop = tf.nn.dropout(self.outputs, keep_prob=KEEP_PROP_DROPOUT)
        full_1_drop = self.outputs
        # Linear activation, using rnn inner loop last output
        return tf.matmul(full_1_drop, self.weights['out']) + self.biases['out']

    # Defines weight variable according to a shape
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # Defines bias variable according to a shape
    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # Defines conv 2d layer
    def conv2d(self,input_x, weights):
        return tf.nn.conv2d(input_x, weights, strides=[1, 1, 1, 1], padding='SAME')

    # 2x2 pixel max pooling
    def max_pool_2x2(self,x, name):
        with tf.name_scope(name):
            max_pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            max_pool_reshape = tf.reshape(max_pool, shape=[-1, 2, 2, 1], name='maxPoolReshaped')
            tf.summary.image('max_pool_2x2', max_pool_reshape, max_outputs=16)
        return max_pool

    # convolution layer
    def conv_layer(self,input_x, shape, name):
        with tf.name_scope(name):
            weights = self.weight_variable(shape)
            biases = self.bias_variable([shape[3]])
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
            weights_reshaped = tf.reshape(weights, shape=[-1, shape[0], shape[1], 1], name='w_reshaped')
            tf.summary.image('filter', weights_reshaped, max_outputs=16)
        return tf.nn.relu(self.conv2d(input_x, weights) + biases)

    def full_layer(self,input, size, name):
        with tf.name_scope(name):
            in_size = int(input.get_shape()[1])
            weights = self.weight_variable([in_size, size])
            biases = self.bias_variable([size])
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("biases", biases)
        return tf.matmul(input, weights) + biases

    def CNN(self):
        # Image input
        x_input = tf.reshape(self.X_cnn, [-1, Board.NUM_ROWS, Board.NUM_COLUMNS, len(Board.EMPTY_CELL)])
        # filter_summary = tf.summary.image('imsum',x_image,max_outputs=2)

        # Convolution Layer 1
        conv_1 = self.conv_layer(x_input,
                            shape=[X_CELL_FILTER_CONV_1, Y_CELL_FILTER_CONV_1, 2,
                                   NUM_FILTER_CONV_1_OUT],
                            name='convolution_layer_1')
        conv_1_pool = self.max_pool_2x2(conv_1, name='convolution_1_pool')

        # Convolution Layer 2
        conv_2 = self.conv_layer(conv_1_pool,
                            shape=[X_CELL_FILTER_CONV_2, Y_CELL_FILTER_CONV_2, NUM_FILTER_CONV_2_IN,
                                   NUM_FILTER_CONV_2_OUT],
                            name='convolution_layer_2')
        conv_2_pool = self.max_pool_2x2(conv_2, name='convolution_2_pool')

        # Flatten layer
        conv_2_flat = tf.reshape(conv_2_pool, [-1, 2*2*20])

        full_1 = tf.nn.relu(self.full_layer(conv_2_flat, 1024, name='fully_layer_1'))
        full_1_drop = tf.nn.dropout(full_1, keep_prob=0.7)

        # Fully Connected outputlayer
        y_conv = self.full_layer(full_1_drop, self.num_classes, name='y_conv')
        return y_conv

    # Create a weight array that gives weight to every step
    def discount_rewards(self, rewards, gamma):
        exponent = len(rewards)-1
        for i in range(0, len(rewards)):
            rewards[i] = rewards[i] * pow(gamma, exponent)
            exponent -= 1

    def set_logger(self, logger):
        self.logger = logger

    # Predict next "winning" game state on the board matrix using RNN
    def predict_rnn(self, x_input):
        x = np.array(x_input)
        x = x_input.reshape(1, self.timesteps, self.num_input)
        predicted = self. sess.run([self.prediction_rnn], feed_dict={self.X_rnn: x})
        #predicted = np.array(predicted).reshape(Board.NUM_COLUMNS)
        return predicted

    # Predict next "winning" game state on the board matrix using RNN
    def predict_cnn(self, x_input):
        x = np.array(x_input)
        x = np.array(x).reshape(1, Board.NUM_ROWS,Board.NUM_COLUMNS, len(Board.EMPTY_CELL))
        predicted = self.sess.run([self.prediction_cnn], feed_dict={self.X_cnn: x})
        predicted = np.array(predicted).reshape(Board.NUM_COLUMNS)
        return predicted

    # Randomly shuffle data. Used for stochastic gradient descent
    def shuffle_batch(self,batch_x,batch_y,rewards):
        locked_data = list(zip(batch_x, batch_y, rewards))
        shuffle(locked_data)
        batch_x, batch_y, rewards = zip(*locked_data)
        return batch_x, batch_y, rewards

    # Getting rewards according to the winner. Winning: positive , Loosing: negative
    # todo add padding
    def get_rewards(self, winner_char, length):
        if winner_char == Board.X_OCCUPIED_CELL:
            rewards = 1 * np.ones(shape=[length, 1])
        elif winner_char == Board.O_OCCUPIED_CELL:
            rewards = -1 * np.ones(shape=[length, 1])
        elif winner_char == Board.EMPTY_CELL:
            rewards = -1 * np.ones(shape=[length, 1])
            #rewards[length-1] = 2 * rewards[length-1]

        return rewards

    def _train_cnn(self):
        self.sess.run(self.init)
        # Start training
        step_index = 0
        epoch = 1
        while epoch > 0:
            for log in self.logger.logs:
                rewards = self.get_rewards(log.winner_char, log.length - 1)
                self.discount_rewards(rewards, 0.99)  # use 1 since a discrete number of steps result in an endstate
                self.batch_size = 1
                for i in range(0, log.length-1):
                    batch_x = log.get_state(i)
                    batch_y, cell_type = log.get_next_step(i)
                    batch_x = np.array(batch_x).reshape((self.batch_size, Board.NUM_ROWS, Board.NUM_COLUMNS, len(Board.EMPTY_CELL)))
                    batch_y = np.array(batch_y).reshape((self.batch_size, self.num_classes))
                    reward = np.array(rewards[i]).reshape((self.batch_size))
                    # Only train if next step is done by correct player
                    if cell_type == Board.X_OCCUPIED_CELL:
                        # Run optimization op (backprop)
                        self.sess.run(self.train_operation, feed_dict={self.X_cnn: batch_x, self.Y_cnn: batch_y, self.Rewards: reward})
                        if step_index % self.display_step == 0 or step_index == 1:
                            # Calculate batch loss and accuracy
                            loss, acc = self.sess.run([self.loss_operation, self.accuracy],
                                                      feed_dict={self.X_cnn: batch_x, self.Y_cnn: batch_y, self.Rewards: reward})
                            print("Step " + str(step_index) + ", Minibatch Loss= " + \
                                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                                  "{:.3f}".format(acc))
                            x = np.array(batch_x[0]).reshape(1, self.timesteps, self.num_input)
                            y = np.array(batch_y[0])
                            predicted = self.predict_cnn(x)
                            print(predicted)
                            print(np.argmax(predicted))
                            print(y)
                            print(np.argmax(y))
                        step_index += 1
            epoch -= 1
        print("Optimization Finished: " + str(epoch))

    def _train(self, log):
        self.batch_size = 1
        batch_x = Log_Analyzer(log).get_all_actions(padding=True)[0:-1:2]  #  player one -> every second action
        batch_y = Log_Analyzer(log).get_all_actions(padding=True)[1::2] #  player second -> every second action
        self.actual_game_actions = np.count_nonzero(batch_y == 1)
        batch_x = np.array(batch_x).reshape((self.batch_size, self.timesteps, self.num_input))
        batch_y = np.array(batch_y).reshape((self.batch_size, self.timesteps, self.num_input))
        rewards = self.get_rewards(log.winner_char, self.actual_game_actions)
        self.discount_rewards(rewards, 0.99)  # use 1 since a discrete number of steps result in an endstate
        reward = np.array(rewards).reshape(self.actual_game_actions)
        # Only train if next step is done by correct player
        # Run optimization op (backprop)
        self.sess.run(self.train_operation,
                      feed_dict={self.X_rnn: batch_x,
                                 self.Y_rnn: batch_y,
                                 self.Rewards: reward})
        if self.global_step % self.display_step == 0 or self.global_step == 1:
            # Calculate batch loss and accuracy
            loss, acc = self.sess.run([self.loss_operation, self.accuracy],
                                      feed_dict={self.X_rnn: batch_x, self.Y_rnn: batch_y, self.Rewards: reward})
            print("Step " + str(self.global_step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            x = np.array(batch_x[0]).reshape(1, self.timesteps, self.num_input)
            y = np.array(batch_y[0])
            predicted = self.predict_rnn(x)
            #print(predicted)
            for prediction in predicted[0]:
                print(np.argmax(prediction))
            print(y)
            print(np.argmax(y))

    # Train network using a specific log
    def train(self, log):
        if log is None:
            self.sess.run(self.init)
            # Start training
            epoch = 1
            while epoch > 0:
                for log in self.logger.logs:
                    self._train(log)
                    epoch -= 1
                    self.global_step += 1
            print("Optimization Finished: " + str(epoch))

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


model = AI('rnn')
model.train(log=None)
#model.save(model.global_step)
#model.load('my_test_model-1000')
#print("Finished")


