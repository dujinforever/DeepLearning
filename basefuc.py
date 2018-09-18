# -*- coding: utf-8 -*-
"""
Created on Sat Aug 2 14:42:49 2018
class MyGan necessary function 
@author: Dujin
"""
import numpy as np
import tensorflow as tf
import os, gzip
import matplotlib.pyplot as plt
import scipy.misc
def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)

def load_mnist(dataset_name):
        data_dir = dataset_name

        def extract_data(filename, num_data, head_size, data_size):
            with gzip.open(filename) as bytestream:
                bytestream.read(head_size)
                buf = bytestream.read(data_size * num_data)
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
            return data

        data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
        trX = data.reshape((60000, 28, 28, 1))

        data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
        trY = data.reshape((60000))

        data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
        teX = data.reshape((10000, 28, 28, 1))

        data = extract_data(data_dir + '/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
        teY = data.reshape((10000))

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int)

        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

        y_vec = np.zeros((len(y), 10), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0
        return X / 255., y_vec
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope,reuse = False):
        matrix = tf.get_variable("Matrix1", [shape[1], output_size], tf.float32,tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias1", [output_size],initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
        
def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, name="deconv2d", stddev=0.02, with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv
        
def conv2d(input_, output_dim, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)
def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)
def inverse_transform(images):
    return (images+1.)/2.	
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)
def merge_images(images, size):
    return inverse_transform(images)
