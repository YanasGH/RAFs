#!/usr/bin/env python
# coding: utf-8

import jax.numpy as np
import numpy
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import BaggingRegressor
from tensorflow.keras.layers import Dense, Input, Lambda, add
from tensorflow.keras.models import Sequential, Model
import tensorflow
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution() 


def rmse(y_pred,y_test, verbose=True):
    '''
    y_pred : tensor 
    y_test : tensor having the same shape as y_pred
    '''
    y_pred = y_pred.reshape(len(y_pred),-1)
    y_test = y_test.reshape(len(y_test),-1)
    ## element wise square
    square = tf.square(y_pred - y_test)## preserve the same shape as y_pred.shape
    ## mean across the final dimensions
    ms = tf.reduce_mean(square)
    rms = tf.math.sqrt(ms)
    return(rms)

def gaussian_nll(y_pred, sigma, y_true):
    '''sigma is actualy sigma squared (aka variance)'''
    y_pred = y_pred.reshape(len(y_pred),-1)
    sigma = sigma.reshape(len(sigma), -1)
    y_true = y_true.reshape(len(y_true), -1)

    y_pred = tf.cast(y_pred, dtype='float32')
    sigma = tf.cast(sigma, dtype='float32')
    y_true = tf.cast(y_true, dtype='float32')
    return tf.reduce_mean(0.5*tf.log(sigma) + 0.5*tf.divide(tf.square(y_true - y_pred), sigma)) + 1e-6

def std_error(y_pred):
    return np.std(y_pred, ddof=1) / np.sqrt(np.size(y_pred))

def report_res(dname, test_targets, method, means, stds):
    method_means = means
    method_stds = stds

#     print(method)
    print()
    print("Results:")
    print("RMSE: ", np.round(rmse(method_means, test_targets).eval(session=tf.compat.v1.Session()), 2))
    print("NLL:", np.round(gaussian_nll(method_means, method_stds**2, test_targets).eval(session=tf.compat.v1.Session()), 2))
    print("SE", np.round(std_error(method_means), 2))
    print("95% CI:", 1.96*std_error(method_means))
    print("STD", np.std(method_means, ddof=1))



