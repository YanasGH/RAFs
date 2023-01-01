#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import os
import jax.numpy as np
import numpy
from collections import namedtuple
from jax import random
import argparse
import pandas as pd

Data = namedtuple('Data', ['inputs', 'targets'])

def main():
    cmdline_parser = argparse.ArgumentParser(description='Script for running ensembles on chosen dataset.')
    cmdline_parser.add_argument('--dataset', help='Choose the dataset from the list: "Superconductivity", "Popularity"', default='Superconductivity')
    args = cmdline_parser.parse_args()
    dname = args.dataset
    return dname
dname = main()  

print("----- RAFs Ensemble vs RP-param utilizing more complex architecture on the high-dimensional large dataset {}: -----".format(dname))

if dname in ["Popularity", "Superconductivity"]:
    if dname == "Popularity":
        data = pd.read_csv(os.getcwd() + '/init/data/OnlineNewsPopularity.csv')
        noise_scale = 1e-1
        X = np.array(data[data.columns[[2,3,4,5,6,7,8,11,12,21,22,23,24,25,26,27,28,29, 30, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]]])
        y = np.array(data[data.columns[60]])

        train_xs = X[0:17840,:]
        train_ys = y[0:17840,].reshape(-1,1)
        test_xs = X[17840:,:]
        test_ys = y[17840:,].reshape(-1,1)

        train = Data(inputs = train_xs, targets = train_ys)
        test = Data(inputs = test_xs, targets = test_ys)

    else:
        data = pd.read_csv(os.getcwd() + '/init/data/superconduct.csv')
        noise_scale = 1e-1
        X = np.array(data[data.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48, 51, 52, 53, 54, 55, 56, 57, 58, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 78]]])
        y = np.array(data[data.columns[-1]])

        train_xs = X[0:9569,:]
        train_ys = y[0:9569:,].reshape(-1,1)
        test_xs = X[9569:,:]
        test_ys = y[9569:,].reshape(-1,1)

    #The training and testing data is in the form of a named tuple
    train = Data(inputs = train_xs, targets = train_ys)
    test = Data(inputs = test_xs, targets = test_ys)
    train_points = len(train_xs)
    test_points = len(test_xs)


    print("---------- RAFs Ensemble ----------")
    import rafs_complex_arch
    print("---------- RP-param ----------")
    import rpparam_complex_arch

else:
    print("{} is not implemented. Please choose Popularity or Superconductivity.".format(dname))