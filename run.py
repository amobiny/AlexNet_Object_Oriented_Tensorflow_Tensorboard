
"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     6/1/2017
Comments:
**********************************************************************************
"""

from datetime import datetime
import time
import h5py
import numpy as np
import tensorflow as tf
from utils import *
from AlexNet import Alexnet
import sys
import os

now = datetime.now()
logs_path = "./graph/" + now.strftime("%Y%m%d-%H%M%S")
save_dir = './checkpoints/'

image_size = 28
num_classes = 10
num_channels = 1
num_epochs = 100
batch_size = 128
display = 100

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = mnist.train.images, mnist.train.labels,\
        mnist.validation.images, mnist.validation.labels, mnist.test.images, mnist.test.labels

X_train, _ = reformat(X_train, Y_train, image_size, num_channels, num_classes)
X_valid, _ = reformat(X_valid, Y_valid, image_size, num_channels, num_classes)
X_test, _ = reformat(X_test, Y_test, image_size, num_channels, num_classes)
print('Training set', X_train.shape, Y_train.shape)
print('Validation set', X_valid.shape, Y_valid.shape)
print('Test set', X_test.shape, Y_test.shape)

# Creating the alexnet model
model = Alexnet(num_classes, image_size, num_channels)
model.inference().accuracy_func().loss_func().train_func()

# Saving the best trained model (based on the validation accuracy)
saver = tf.train.Saver()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'best_validation')
best_validation_accuracy = 0

acc_b_all = loss_b_all = np.array([])
sum_count = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized")
    merged = tf.summary.merge_all()
    batch_writer = tf.summary.FileWriter(logs_path + '/batch/', sess.graph)
    valid_writer = tf.summary.FileWriter(logs_path + '/valid/')
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        print('-----------------------------------------------------------------------------')
        print('Epoch: {}'.format(epoch+1))
        X_train, Y_train = randomize(X_train, Y_train)
        step_count = int(len(X_train)/batch_size)
        for step in range(step_count):
            start = step * batch_size
            end = (step + 1) * batch_size
            X_batch, Y_batch = get_next_batch(X_train, Y_train, start, end)
            # X_batch = random_rotation_2d(X_batch, 20.0)
            feed_dict_batch = {model.x: X_batch, model.y: Y_batch, model.keep_prob: 0.5}

            _, acc_b, loss_b = sess.run([model.train_op, model.accuracy, model.loss], feed_dict=feed_dict_batch)
            acc_b_all = np.append(acc_b_all, acc_b)
            loss_b_all = np.append(loss_b_all, loss_b)

            if step % display == 0:
                mean_acc = np.mean(acc_b_all)
                mean_loss = np.mean(loss_b_all)
                print("Step {0}, training loss: {1:.5f}, training accuracy: {2:.01%}".format(step, mean_loss, mean_acc))
                summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Accuracy', simple_value=mean_acc)])
                batch_writer.add_summary(summary_tr, sum_count * display)
                summary_tr = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=mean_loss)])
                batch_writer.add_summary(summary_tr, sum_count * display)
                summary = sess.run(merged, feed_dict=feed_dict_batch)
                batch_writer.add_summary(summary, sum_count * display)
                sum_count += 1
                acc_b_all = loss_b_all = mean_acc = mean_loss = y_pred_all = y_true_all = np.array([])

        feed_dict_val = {model.x: X_valid, model.y: Y_valid, model.keep_prob: 1}
        acc_valid, loss_valid = sess.run([model.accuracy, model.loss], feed_dict=feed_dict_val)
        summary_valid = tf.Summary(value=[tf.Summary.Value(tag='Accuracy', simple_value=acc_valid)])
        valid_writer.add_summary(summary_valid, sum_count * display)
        if acc_valid > best_validation_accuracy:
            # Update the best-known validation accuracy.
            best_validation_accuracy = acc_valid
            best_epoch = epoch
            # Save all variables of the TensorFlow graph to file.
            saver.save(sess=sess, save_path=save_path)
            # A string to be printed below, shows improvement found.
            improved_str = '*'
        else:
            # An empty string to be printed below.
            # Shows that no improvement was found.
            improved_str = ''
        print("Epoch {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}{3}"
              .format(epoch + 1, loss_valid, acc_valid, improved_str))


