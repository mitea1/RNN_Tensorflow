import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from src.Board import Board
import pickle
import os

LOCATION = os.getcwd()
FILE_NAME_LOGGER = LOCATION + '/data/log/logger.pkl'

class RNN_Model:

    def __init__(self):
        # Load data
        with open(FILE_NAME_LOGGER, 'rb') as f:
            self.logger = pickle.load(f)

        # Build RNN Graph

        # Training Parameters
        self.learning_rate = 0.01
        self.learning_rate_decay = 0.95
        self.training_steps = 2000
        self.batch_size = 1
        self.display_step = 100

        # Network Parameters
        self.num_input = Board.NUM_COLUMNS * Board.NUM_ROWS * len(Board.EMPTY_CELL)  # Board  dimensions
        self.timesteps = 1  # timesteps
        self.num_hidden = 128  # hidden layer num of features
        self.num_classes = Board.NUM_COLUMNS * Board.NUM_ROWS * len(Board.EMPTY_CELL)  # Board dimension (prediction)

        # tf Graph input
        self.X = tf.placeholder("float", [None, self.timesteps, self.num_input])  # Input of RNN
        self.Y = tf.placeholder("float", [None, self.num_classes])  # Output of RNN
        self.Rewards = tf.placeholder("float", shape=[None])  # Reward input to weight input

        # Define weights and biases
        self.weights = {
            'out': tf.Variable(tf.random_normal([self.num_hidden, self.num_classes]))
        }
        self.biases = {
            'out': tf.Variable(tf.random_normal([self.num_classes]))
        }

        self.logits = self.RNN()
        self.prediction = tf.nn.softmax(self.logits)

        # Define loss and optimizer
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y)
        self.loss_operation = tf.reduce_mean(self.Rewards * self.cross_entropy)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_operation = self.optimizer.minimize(self.loss_operation)

        # Evaluate model (with test logits, for dropout to be disabled)
        self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

    def RNN(self):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, timesteps, n_input)
        # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
        x = tf.unstack(self.X, self.timesteps, 1)

        # Define a rnn cell with tensorflow
        rnn_cell = rnn.BasicRNNCell(self.num_hidden)

        # Get lstm cell output
        outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

    def discount_rewards(self, rewards, gamma):
        exponent = len(rewards)
        for i in range(0, len(rewards)):
            rewards[i] = rewards[i] * pow(gamma, exponent)
            exponent -= 1


    def train(self):
        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(self.init)
            step_index = 0
            data_size = len(self.logger.logs)
            for log in self.logger.logs:
                if log.winner_char[0] == 1:
                    rewards = 1 * np.ones(shape=[log.length - 1, 1])
                else:
                    rewards = 1 * np.ones(shape=[log.length - 1, 1])
                self.discount_rewards(rewards, 0.95)

                for step_index in range(0, log.length-1):
                    batch_x, batch_y = log.get_state(step_index), log.get_next_step(step_index)
                    # Reshape data to get 28 seq of 28 elements
                    batch_x = batch_x.reshape((self.batch_size, self.timesteps, self.num_input))
                    batch_y = batch_y.reshape((self.batch_size, self.num_input))
                    # Run optimization op (backprop)
                    sess.run(self.train_operation, feed_dict={self.X: batch_x, self.Y: batch_y, self.Rewards: rewards[step_index]})
                    if step_index % self.display_step == 0 or step_index == 1:
                        # Calculate batch loss and accuracy
                        loss, acc = sess.run([self.loss_operation, self.accuracy],
                                             feed_dict={self.X: batch_x, self.Y: batch_y, self.Rewards: rewards[step_index]})
                        print("Step " + str(step_index) + ", Minibatch Loss= " + \
                              "{:.4f}".format(loss) + ", Training Accuracy= " + \
                              "{:.3f}".format(acc))
                        #x = np.array(batch_x)
                        #y = np.array(batch_y)
                        #predicted = sess.run([self.prediction], feed_dict={self.X: x})
                        #print(predicted)
                        #print(y)
                    step_index += 1
            print("Optimization Finished!")


model = RNN_Model()
model.train()