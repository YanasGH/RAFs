#!/usr/bin/env python
# coding: utf-8

# get_ipython().system('pip install --upgrade pip')
# get_ipython().system('pip install --upgrade jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html')
# get_ipython().system('pip install -q git+https://www.github.com/google/neural-tangents')

# pip install --upgrade pip # TODO: add in txt file
# pip install jax jaxlib --upgrade # TODO: add in txt file
# pip install neural-tangents  # TODO: add in txt file

# from neural_tangents_main import neural_tangents as nt
# print('----------------ok')
import sys, os
import jax.numpy as np
import tensorflow
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution() 
from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad, vmap

import neural_tangents as nt
from neural_tangents import stax
from collections import namedtuple

import random as rdm
from collections import OrderedDict
from datetime import datetime
from math import floor

from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy
from sklearn.preprocessing import StandardScaler

import keras
import scipy.stats as scipy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, concatenate, Input, Embedding
from tensorflow.keras.layers import Reshape, Concatenate, BatchNormalization, Dropout, Add, Lambda
from tensorflow.keras.layers import add
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.ensemble import BaggingRegressor
from copy import deepcopy
from random import randint

import functools
from IPython.display import set_matplotlib_formats
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('pdf', 'svg')
import matplotlib
import seaborn as sns

sns.set(font_scale=1.3)
sns.set_style("darkgrid", {"axes.facecolor": ".95"})

import matplotlib.pyplot as plt

def format_plot(x=None, y=None):
    ax = plt.gca()
    if x is not None:
        plt.xlabel(x, fontsize=20)
    if y is not None:
        plt.ylabel(y, fontsize=20)

def finalize_plot(shape=(1, 1)):
    plt.gcf().set_size_inches(6.785, 6.785)
#     (shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
#     shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
    plt.tight_layout()
legend = functools.partial(plt.legend, fontsize=10)



def plot_fn(train, test, *fs):
    train_xs, train_ys = train.inputs, train.targets
    plt.plot(train_xs, train_ys, 'ro', markersize=10, label='train')

    if test != None:
        test_xs, test_ys = test
        plt.plot(test_xs, test_ys, 'k--', linewidth=3, label='$f(x)$')
        for f in fs:
            plt.plot(test_xs, f(test_xs), '-', linewidth=3)

    #plt.xlim([-np.pi, np.pi])
    plt.xlim([-1.715,1.106])
    plt.ylim([-1.5, 1.5])

    format_plot('$x$', '$f$')


def loss_fn(predict_fn, ys, t, xs=None):
    mean, cov = predict_fn(t=t, get='ntk', x_test=xs, compute_cov=True)
    mean = np.reshape(mean, mean.shape[:1] + (-1,))
    var = np.diagonal(cov, axis1=1, axis2=2)
    ys = np.reshape(ys, (1, -1))
    mean_predictions = 0.5 * np.mean(ys ** 2 - 2 * mean * ys + var + mean ** 2,
                                       axis=1)

    return mean_predictions


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

def report_res(dname, test, test_targets, method, means, stds, scaler_X, scaler_y):
    if dname == 'Boston housing':
        pred_dict = {'rooms_pre': test.inputs[:,0],
                  'y': test.targets[:,0],
                  'y_mean_pre': means,
                  'y_upper_pre': means + 2*stds,
                  'y_lower_pre': means - 2*stds
                  }
        pred_df = pd.DataFrame(data=pred_dict)
        pred_df_sorted = pred_df.sort_values(by='rooms_pre')
        pred_df_sorted['rooms'] = scaler_X.inverse_transform(numpy.array(pred_df_sorted['rooms_pre']).reshape(-1, 1))
        pred_df_sorted['y_mean'] = scaler_y.inverse_transform(numpy.array(pred_df_sorted['y_mean_pre']).reshape(-1, 1))
        pred_df_sorted['y_upper'] = scaler_y.inverse_transform(numpy.array(pred_df_sorted['y_upper_pre']).reshape(-1, 1))
        pred_df_sorted['y_lower'] = scaler_y.inverse_transform(numpy.array(pred_df_sorted['y_lower_pre']).reshape(-1, 1))

        method_means = np.array(pred_df_sorted['y_mean'].to_list())
        method_stds = (np.array(pred_df_sorted['y_upper'].to_list()) - np.array(pred_df_sorted['y_mean'].to_list()))/2

    else:
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
    
    
def viz_one_d(dname, train, test, method, method_means, method_stds, predictions, saveviz):
    '''Plot the results of the datasets He et al. and Forrester et al. 
     dname: name of the dataset. There are two options: "He" or "Forrester"
     test: test data in the form of a named tuple 
     method: method name
     method_means: aggregated output (array)
     method_stds: stds of the method (array)
     saveviz: True or False, depending on whether the plot needs to be saved
    '''
    if dname in ["He", "Forrester"]:
        plt.subplot(1, 1, 1) #2,2,1
        plot_fn(train, test)#(train, test)

        try:
            ntkgp_moments = predictions['NTKGP analytic']
            ntkgp_means = ntkgp_moments.mean
            ntkgp_stds = ntkgp_moments.standard_deviation
            gp_plot, = plt.plot(test.inputs, ntkgp_means, 'red', linewidth = 3, alpha = 0.5)
            plt.fill_between(
                np.reshape(test.inputs, (-1,)),
                  ntkgp_means - 2 * ntkgp_stds,
                  ntkgp_means + 2 * ntkgp_stds,
                  color='red',
                  alpha = 0.3)
        except:
            print('Run cell "NTKGP-param and Deep Ensemble", as the GP is included there')
            gp_plot = None

        method_plot, = plt.plot(test.inputs, method_means, 'blue', linewidth = 2, alpha = 0.5)
        plt.fill_between(np.reshape(test.inputs, (-1,)),
                np.reshape(method_means - 2 * method_stds, (-1,)),
                np.reshape(method_means + 2 * method_stds, (-1,)),
                color='blue',
                alpha = 0.3)

        legend((gp_plot,method_plot,), ["NTKGP analytic", method], loc="lower center", fontsize="large")

        if dname == "He":
            plt.xlim([-6, 6])
            plt.ylim([-6, 6])
            format_plot('$x$', '$f$')
            finalize_plot((1,1))

        else:
            plt.xlim([0, 1])
            plt.ylim([-12, 18])
            format_plot('$x$', '$f$')
            finalize_plot((1,1))

        if saveviz:
            plt.savefig("{}_{}.png".format(dname, method), dpi = 300)
        plt.clf()
    else:
        print("Only the results on He et al. and Forrester et al. can be plotted as those are one-dimensional datasets.")





