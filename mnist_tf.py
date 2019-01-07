#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import ipdb

data_dir = './mnist_dataset'
mnist = input_data.read_data_sets(data_dir, one_hot=True)
# ipdb.set_trace()

if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, w) + b

    y_ = tf.placeholder(tf.int64, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    sess.close()
