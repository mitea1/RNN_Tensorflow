import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


three_series = [number * 3 for number in range(1, 11)]

three_series_to_index = {num:i for i,num in enumerate(three_series)}
index_to_three_series = {i:num for i,num in enumerate(three_series)}

# RNN Parameters
element_size = 10
time_steps = 3
num_classes = 10
batch_size = 3
hidden_layer_size = 128

# Input & Output of NN
_inputs = tf.placeholder(tf.float32, shape=[None, time_steps, element_size], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='inputs')

# TensorFlow built-in functions
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

# Weights and biases
Weights_linear = tf.Variable(tf.truncated_normal([hidden_layer_size, num_classes], mean=0, stddev=.01))
biases_linear = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))


def get_linear_layer(linear_input_vector):
    return tf.matmul(linear_input_vector, Weights_linear) + biases_linear

def get_batch(batch_size):
    batch_x = np.zeros((3, num_classes))
    batch_y = np.zeros((1, num_classes))
    for i in range(0, 3):
        batch_x[i][i] = 1
    batch_y[0][i+1] = 1

    return batch_x, batch_y

last_rnn_output = outputs[:, -1, :]
final_output = get_linear_layer(last_rnn_output)

softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output, labels=y)
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

# Check if prediction is correct
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_output, 1))
# Calculate accuracy
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100


# Cut test image into slices
#test_data = mnist.test.images[:batch_size].reshape((-1,time_steps,element_size))
#test_label = mnist.test.labels[:batch_size]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(1001):

    batch_x, batch_y = get_batch(batch_size)
    batch_x = batch_x.reshape((1, time_steps, element_size))
    sess.run(train_step, feed_dict={_inputs: batch_x, y: batch_y})
    if i % 20 == 0:
        acc = sess.run(accuracy, feed_dict={_inputs: batch_x, y: batch_y})
        loss = sess.run(cross_entropy, feed_dict={_inputs: batch_x, y: batch_y})
        print("Iter " + str(i) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        #print("Testing Accuracy:",sess.run(accuracy, feed_dict={_inputs: test_data, y: test_label}))