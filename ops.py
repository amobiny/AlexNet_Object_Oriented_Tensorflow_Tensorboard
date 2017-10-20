"""
Copyright 2017-2022 Department of Electrical and Computer Engineering
University of Houston, TX/USA
**********************************************************************************
Author:   Aryan Mobiny
Date:     6/1/2017
Comments: Includes functions for defining the AlexNet layers
**********************************************************************************
"""

import tensorflow as tf


def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name, dtype=tf.float32,
                           shape=shape, initializer=initer)


def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initial bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name, dtype=tf.float32,
                           initializer=initial)


def fc_layer(bottom, out_dim, name, is_train=True, batch_norm=False, add_reg=False, use_relu=True):
    """
    Creates a fully-connected layer
    :param bottom: input from previous layer
    :param out_dim: number of hidden units in the fully-connected layer
    :param name: layer name
    :param is_train: boolean to differentiate train and test (useful when applying batch normalization)
    :param batch_norm: boolean to add the batch normalization layer (or not)
    :param add_reg: boolean to add norm-2 regularization (or not)
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    in_dim = bottom.get_shape()[1]
    with tf.variable_scope(name):
        weights = weight_variable(name, shape=[in_dim, out_dim])
        tf.summary.histogram('W', weights)
        biases = bias_variable(name, [out_dim])
        layer = tf.matmul(bottom, weights)
        if batch_norm:
            layer = batch_norm_wrapper(layer, is_train)
        layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def conv_2d(inputs, filter_size, stride, num_filters, name,
            is_train=True, batch_norm=False, add_reg=False, use_relu=True):
    """
    Create a 2D convolution layer
    :param inputs: input array
    :param filter_size: size of the filter
    :param stride: filter stride
    :param num_filters: number of filters (or output feature maps)
    :param name: layer name
    :param is_train: boolean to differentiate train and test (useful when applying batch normalization)
    :param batch_norm: boolean to add the batch normalization layer (or not)
    :param add_reg: boolean to add norm-2 regularization (or not)
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    num_in_channel = inputs.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        shape = [filter_size, filter_size, num_in_channel, num_filters]
        weights = weight_variable(name, shape=shape)
        tf.summary.histogram('W', weights)
        biases = bias_variable(name, [num_filters])
        layer = tf.nn.conv2d(input=inputs,
                             filter=weights,
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        print('{}: {}'.format(layer.name, layer.get_shape()))
        if batch_norm:
            layer = batch_norm_wrapper(layer, is_train)
        layer += biases
        if use_relu:
            layer = tf.nn.relu(layer)
        if add_reg:
            tf.add_to_collection('weights', weights)
    return layer


def flatten_layer(layer):
    """
    Flattens the output of the convolutional layer to be fed into fully-connected layer
    :param layer: input array
    :return: flattened array
    """
    with tf.variable_scope('Flatten_layer'):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat


def max_pool(x, ksize, stride, name):
    """
    Create a max pooling layer
    :param x: input to max-pooling layer
    :param ksize: size of the max-pooling filter
    :param stride: stride of the max-pooling filter
    :param name: layer name
    :return: The output array
    """
    maxpool = tf.nn.max_pool(x,
                             ksize=[1, ksize, ksize, 1],
                             strides=[1, stride, stride, 1],
                             padding="SAME",
                             name=name)
    print('{}: {}'.format(maxpool.name, maxpool.get_shape()))
    return maxpool


def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer"""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def dropout(x, keep_prob):
    """
    Create a dropout layer
    :param x: input to dropout layer
    :param keep_prob: dropout rate (e.g.: 0.5 means keeping 50% of the units)
    :return: the output array
    """
    return tf.nn.dropout(x, keep_prob)


def batch_norm_wrapper(inputs, is_training, decay=0.999, epsilon=1e-3):
    """
    creates a batch normalization layer
    :param inputs: input array
    :param is_training: boolean for differentiating train and test
    :param decay:
    :param epsilon:
    :return: normalized input
    """
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if len(inputs.get_shape().as_list()) == 4:  # For convolutional layers
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        else:                                       # For fully-connected layers
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)
