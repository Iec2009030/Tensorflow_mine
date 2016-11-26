#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 16:46:14 2016

@author: Nitin Bansal
Working on MNIST dataset for Classification of Digits 
Accuracy obtained up to 98%
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

with tf.Session() as sess:
    x = tf.placeholder("float",shape =[None,784])
    y_ = tf.placeholder("float", shape = [None,10])
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x,W) + b)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy)

    init_op = tf.initialize_all_variables()
    init_op.run()

    for i in range(1000):
        batch_x,batch_y = mnist.train.next_batch(100)
        train_step.run(feed_dict={x: batch_x, y_: batch_y})


        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
        print (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
   
