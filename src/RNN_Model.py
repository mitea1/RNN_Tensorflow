from random import Random, shuffle

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from src.Board import Board
import pickle
import os

LOCATION = os.getcwd()
FILE_NAME_LOGGER = LOCATION + '/data/log/loggerSat Jan 27 15:57:53 2018.pkl'
FILE_PATH_SESSION = LOCATION + '/data/session/'
FILE_NAME_BIASES = LOCATION + '/data/model/rnn_biases.pkl'

class RNN_Model:

    def __init__(self):
        # Load data
        with open(FILE_NAME_LOGGER, 'rb') as f:
            self.logger = pickle.load(f)

        # Build RNN Graph

        # Training Parameters
        self.learning_rate = 0.01
        self.learning_rate_decay = 0.99
        self.training_steps = 2000
        self.batch_size = 1
        self.display_step = 9

        # Network Parameters
        self.num_input = Board.NUM_COLUMNS * len(Board.EMPTY_CELL)  # Board  dimensions
        self.timesteps = Board.NUM_ROWS  # timesteps
        self.num_hidden = 1024  # hidden layer num of features
        self.num_classes = Board.NUM_COLUMNS  # Board dimension (prediction)

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.timesteps, self.num_input])  # Input of RNN
        self.Y = tf.placeholder("float", [None, self.num_classes])  # Output of RNN
        self.Rewards = tf.placeholder("float", shape=[None])  # Reward input to weight input

        # Define weights and biases
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.num_hidden, self.num_classes], mean=0, stddev=0.01))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.num_classes], mean=0, stddev=0.01))
        }

        self.logits = self.RNN()
        self.prediction = tf.nn.relu(self.logits)

        # Define loss and optimizer
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        self.loss_operation = tf.reduce_mean(self.Rewards * self.cross_entropy)
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,decay=self.learning_rate_decay)
        self.train_operation = self.optimizer.minimize(self.loss_operation)

        # Evaluate model (with test logits, for dropout to be disabled)
        self.correct_prediction = tf.equal(tf.argmax(self.prediction,1), tf.argmax(self.Y,1))
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
        x = tf.unstack(self.X, self.timesteps, 1)

        # Define a rnn cell with tensorflow
        #rnn_cell = rnn.BasicRNNCell(self.num_hidden)
        rnn_cell = rnn.LSTMCell(self.num_hidden)

        # Get RNN cell output
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']



    # Create a weight array that gives weight to every step
    def discount_rewards(self, rewards, gamma):
        exponent = len(rewards)
        for i in range(0, len(rewards)):
            rewards[i] = rewards[i] * pow(gamma, exponent)
            exponent -= 1

    def set_logger(self, logger):
        self.logger = logger

    # Predict next "winning" game state on the board matrix
    def predict(self, x_input):
        x = np.array(x_input)
        x = np.array(x).reshape(1, self.timesteps, self.num_input)
        predicted = self. sess.run([self.prediction], feed_dict={self.X: x})
        predicted = np.array(predicted).reshape(Board.NUM_COLUMNS)
        return predicted

    # Randomly shuffle data. Used for stochastic gradient descent
    def shuffle_batch(self,batch_x,batch_y,rewards):
        locked_data = list(zip(batch_x, batch_y, rewards))
        shuffle(locked_data)
        batch_x, batch_y, rewards = zip(*locked_data)
        return batch_x, batch_y, rewards

    # Getting rewards according to the winner. Winning: positive , Loosing: negative
    def get_rewards(self, winner_char, length):
        if winner_char == Board.X_OCCUPIED_CELL:
            rewards = 1 * np.ones(shape=[length, 1])
        elif winner_char == Board.O_OCCUPIED_CELL:
            rewards = 1 * np.ones(shape=[length, 1])
        return rewards

    def train(self):
        self.sess.run(self.init)
        # Start training
        step_index = 0
        epoch = 1
        while epoch > 0:
            log_length = 1000
            for log in self.logger.logs:
                if log_length == 0:
                    break
                log_length -= 1
            #for i in range(0,100):
                #log = self.logger.logs[i]
                rewards = self.get_rewards(log.winner_char, log.length - 1)
                self.discount_rewards(rewards, 1)  # use 1 since a discrete number of steps result in an endstate
                self.batch_size = 1
                for i in range(0, self.batch_size):
                    batch_x, batch_y = log.get_state(i), log.get_next_step(i)
                    #batch_x, batch_y, rewards = self.shuffle_batch(batch_x, batch_y, rewards)
                    batch_x = np.array(batch_x).reshape((self.batch_size, self.timesteps, self.num_input))
                    batch_y = np.array(batch_y).reshape((self.batch_size, self.num_classes))
                    reward = np.array(rewards[i]).reshape((self.batch_size))
                    # Run optimization op (backprop)
                    self.sess.run(self.train_operation, feed_dict={self.X: batch_x, self.Y: batch_y, self.Rewards: reward})
                    if step_index % self.display_step == 0 or step_index == 1:
                        # Calculate batch loss and accuracy
                        loss, acc = self.sess.run([self.loss_operation, self.accuracy],
                                             feed_dict={self.X: batch_x, self.Y: batch_y, self.Rewards: reward})
                        print("Step " + str(step_index) + ", Minibatch Loss= " + \
                              "{:.4f}".format(loss) + ", Training Accuracy= " + \
                              "{:.3f}".format(acc))
                        x = np.array(batch_x[0]).reshape(1, self.timesteps, self.num_input)
                        y = np.array(batch_y[0])
                        predicted = self.predict(x)
                        print(predicted)
                        print(np.argmax(predicted))
                        print(y)
                        print(np.argmax(y))
                    step_index += 1
            epoch -= 1
        print("Optimization Finished: " + str(epoch))

    # Save the the actual session
    def save(self):
        self.saver.save(self.sess, FILE_PATH_SESSION+'my_test_model', global_step=1000)

    # Load the last saved session
    def load(self, session_name):
        tf.reset_default_graph()
        filename = session_name + '.meta'
        self.saver = tf.train.import_meta_graph(FILE_PATH_SESSION+filename)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(FILE_PATH_SESSION))


model = RNN_Model()
model.train()
#model.save()
#model.load('my_test_model-1000')
#print("Finished")


