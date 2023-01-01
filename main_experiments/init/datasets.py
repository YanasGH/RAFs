#import modules
import os
import jax.numpy as np
import numpy
from collections import namedtuple
from jax import random
import argparse
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def main():
    cmdline_parser = argparse.ArgumentParser(description='Script for running ensembles on chosen dataset.')
    cmdline_parser.add_argument('--dataset', help='Choose the dataset from the list: "He", "Forrester", "Schaffer", "Double pendulum", "Rastrigin", "Ishigami", "Environmental model", "Griewank", "Roos & Arnold", "Friedman", "Planar arm torque", "Sum of powers", "Ackley", "Piston simulation", "Robot arm", "Borehole", "Styblinski-Tang", "PUMA560", "Adapted Welch", "Wing weight", "Boston housing", "Abalone", "Naval propulsion plant", "Forest fire", "Parkinson"', default='He')
    args = cmdline_parser.parse_args()
    dname = args.dataset
    return dname
dname = main()     # python ./init/datasets.py --dataset "He" 


# the training and testing data is in the form of a named tuple
Data = namedtuple(
    'Data',
    ['inputs', 'targets']
)

# supporting functions in the data-generating process:

# Shaffer generating function
def schaffer(x, y):
    f1 = (np.cos(np.sin(abs(x**2 - y**2)))**2 - 0.5)/((1+0.001*(x**2 + y**2))**2) 
    return 0.5 + f1

# double pendulum generating function
def double_pendulum(theta1, theta2):
    L1, L2 = 1, 1
    return L1*np.sin(theta1) + L2*np.sin(theta2)

# environmental model generating function
def env_model(M, D, L, g):
    t = 40.1
    tc = np.array(len(M)*[t])
    s = 1
    I = (g<tc).astype(int)
    p1 = (M/np.sqrt(4*np.pi*D*t))*np.exp(-(s**2)/(4*D*t))
    C = p1 + (M/np.sqrt(4*np.pi*D*(t-g)))*np.exp(-((s-L)**2)/(4*D*(t-g)))*I

    return np.sqrt(4*np.pi)*C

# Griewank generating function
def griewank(xset):
    term1 = np.array(xset.shape[0]*[0])
    term2 = np.array(xset.shape[0]*[1])
    for i in range(xset.shape[0]):
        term1 += (xset[:,i]**2)/4000
        term2 *= np.cos(xset[:,i]/np.sqrt(i+1))

    return term1 - term2 + 1

# Roos & Arnold generating function
def roos_arnold(train_xs):
    d = train_xs.shape[1]
    prod = np.array(train_xs.shape[0]*[0])
    for i in range(d):
        p = abs(4*train_xs[:,i] - 2)
        prod = prod*p

    return prod/2

# Planar arm torque generating function
def planar_arm_torque(train_xs):
    y = []
    for i in range(len(train_xs[:,0])):
        m11 = 0.2083 + 0.125*np.cos(train_xs[i,1])
        m12 = 0.0417+0.0625*np.cos(train_xs[i,1])
        m22 = 0.0417 
        M = np.array([[m11, m12],[m12, m22]])

        c11 = -0.0625*np.sin(train_xs[i,1])*train_xs[i,3]
        c12 = -0.0625*np.sin(train_xs[i,1])*(train_xs[i,2] + train_xs[i,3])
        c21 = 0.0625*np.sin(train_xs[i,1])*train_xs[i,2]
        c22 = 0  #np.array(len(train_xs[:,0])*[0])
        C = np.array([[c11, c12],[c21, c22]])

        vec1 = train_xs[i,4:].T.reshape(2,1)
        vec2 = train_xs[i,2:4].T.reshape(2,1)
        summed = M@vec1 + C@vec2
        y.append(summed.T.tolist()[0])
      
    return np.array(y)
 
# sum of powers generating function
def sum_of_powers(xset):
    term = np.array(xset.shape[0]*[0])
    for i in range(xset.shape[0]):
        term += abs(xset[:,i])**(i+1)
        
    return term

# Ackley generating function
def ackley(xset):   
    a, b, c= 20, 0.2, 2*np.pi
    d = xset.shape[1]
    p1 = a + np.exp(1)
    sum1 = np.array(xset.shape[0]*[0])
    sum2 = np.array(xset.shape[0]*[0])
    for i in range(d):
        sum1 += xset[:,i]**2
        sum2 += np.cos(c*xset[:,i])
    p2 =-a*np.exp(-b*np.sqrt((1/d)*sum1))
    p3 = -np.exp((1/d)*sum2)

    return p1+p2+p3

# piston simulation generating function
def piston(train_xs):
    A = train_xs[:,4]*train_xs[:,1] + 19.62*train_xs[:,0] - (train_xs[:,3]*train_xs[:,2])/train_xs[:,1]
    V = (train_xs[:,1]/(2*train_xs[:,3]))*(np.sqrt(A**2 + (4*train_xs[:,5]*train_xs[:,3]*train_xs[:,2]*train_xs[:,4])/train_xs[:,6])-A)
    C = 2*np.pi*np.sqrt(train_xs[:,0]/(train_xs[:,3] + ((train_xs[:,1]**2)*train_xs[:,4]*train_xs[:,2]*train_xs[:,5])/(train_xs[:,6]*(V**2))))

    return C

# robot arm generating function
def robot_arm(train_xs):
    u = np.zeros(shape=train_xs[:,0].shape) #u
    v = np.zeros(shape=train_xs[:,0].shape) #v
    for i in range(4): 
        L_i = train_xs[:,4+i]
        sum_theta = np.zeros(shape=train_xs[:,0].shape)
        for j in range(i+1):
            sum_theta += train_xs[:,j]
        u += L_i*np.cos(sum_theta)
        v += L_i*np.cos(sum_theta)
    return np.sqrt(u**2 + v**2)

# Borehole generating function
def borehole(rw, r, Tu, Hu, Tl, Hl, L, Kw):
    frac1 = 2 * np.pi * Tu * (Hu-Hl)
    frac2a = 2*L*Tu / (np.log(r/rw)*(rw**2)*Kw)
    frac2b = Tu / Tl
    frac2 = np.log(r/rw) * (1+frac2a+frac2b)

    return frac1 / frac2

# Styblinski_tang generating function
def styblinski_tang(train_xs):
    d = train_xs.shape[1]
    summed = np.array(train_xs.shape[0]*[0])
    for i in range(d):
        summed += train_xs[:,i]**4 - 16*(train_xs[:,i]**2) + 5*train_xs[:,i]

    return summed/2

# adapted Welch et al. generating function
def adapt_welch(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
      return 5*x10/(1.001+x1) + 5*((x4-x2)**2) + x5 + 40*(x9**3) - 5*x1 + 0.08*x3 + 0.25*(x6**2) + 0.03*x7 - 0.09*x8

# wing weight generating function
def wing_weight(S_w, W_fw, A, L, q, lambda1, t_c, N_z, W_dg, W_p):   
    L = L*(np.pi/180)  
    p1 = 0.036*(S_w**0.758)*(W_fw**0.0035)
    p2 = ((A/(np.cos(L)**2))**0.6)*(q**0.006)*(lambda1**0.04)
    p3 = (((100*t_c)/np.cos(L))**(-0.3))*((N_z*W_dg)**0.49)
    p4 = S_w*W_p

    return p1*p2*p3 + p4

# generate data:
def generate_data(dname):
    '''

    dname: The name of the dataset you wish to generate. The options are "He", "Forrester", "Schaffer", "Double pendulum", "Rastrigin", "Ishigami",
         "Environmental model", "Griewank", "Roos & Arnold", "Friedman", "Planar arm torque", "Sum of powers", "Ackley", "Piston simulation", "Robot arm",
         "Borehole", "Styblinski-Tang", "PUMA560", "Adapted Welch", "Wing weight", "Boston housing", "Abalone", "Naval propulsion plant", "Forest fire", "Parkinson"

    '''
    key = random.PRNGKey(10)
    key, x_key, y_key = random.split(key, 3)  # This ensures that the exact same points are always geenrated

    if dname == 'He':
        train_points = 20
        test_points = 50
        noise_scale = 1e-1
        target_fn = lambda x: x*np.sin(x)

        train_xlim = 2
        test_xlim = 6
        half_train_points = train_points // 2
        train_xs_left = random.uniform(x_key, shape = (half_train_points, 1), minval = -train_xlim, maxval = -train_xlim/3)
        train_xs_right = random.uniform(x_key, shape = (half_train_points, 1), minval = train_xlim/3, maxval = train_xlim)

        train_xs = np.concatenate((train_xs_left, train_xs_right))
        train_ys = target_fn(train_xs)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = np.linspace(-test_xlim, test_xlim, test_points)
        test_xs = np.reshape(test_xs, (test_points, 1))
        test_ys = target_fn(test_xs)

    elif dname == 'Forrester':
        train_points = 20
        test_points = 50
        noise_scale = 1e-1
        target_fn = lambda x: ((6*x-2)**2)*np.sin(12*x-4)

        train_xlim = 2
        test_xlim = 6
        half_train_points = train_points // 2
        train_xs_left = random.uniform(x_key, shape = (half_train_points, 1), minval = 0.2, maxval = 0.4)
        train_xs_right = random.uniform(x_key, shape = (half_train_points, 1), minval = 0.65, maxval = 0.85)

        train_xs = np.concatenate((train_xs_left, train_xs_right))
        train_ys = target_fn(train_xs)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = np.linspace(0, 1, test_points)
        test_xs = np.reshape(test_xs, (test_points, 1))
        test_ys = target_fn(test_xs)
  
    elif dname == 'Schaffer':
        train_points = 1000
        test_points = 2500
        noise_scale = 1e-1

        train_xs = random.uniform(x_key, shape = (train_points, 2), minval = -2, maxval = 2)
        train_ys = schaffer(train_xs[:,0], train_xs[:,1]).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = random.uniform( x_key, shape = (test_points, 2), minval = -2.5, maxval = 2.5)
        test_xs = np.reshape(test_xs, (test_points, 2))
        test_ys = schaffer(test_xs[:,0], test_xs[:,1])
  
    elif dname == 'Double pendulum':
        train_points = 1000
        test_points = 2500
        noise_scale = 1e-1

        train_xs = random.uniform(x_key, shape = (train_points, 2), minval = (-2*np.pi)/3, maxval = np.pi/6)
        train_ys = double_pendulum(train_xs[:,0], train_xs[:,1]).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = random.uniform(x_key, shape = (test_points, 2), minval = -np.pi, maxval = np.pi)
        test_xs = np.reshape(test_xs, (test_points, 2))
        test_ys = double_pendulum(test_xs[:,0], test_xs[:,1])
  
    elif dname == 'Rastrigin':
        train_points = 200
        test_points = 500
        noise_scale = 1e-1
        target_fn = lambda x: 30 + ((x[:,0])**2 - 10*np.cos(2*np.pi*x[:,0])**3) + ((x[:,1])**2 - 10*np.cos(2*np.pi*x[:,1])**3) + ((x[:,2])**2 - 10*np.cos(2*np.pi*x[:,2])**3)

        train_xlim = 5.12
        test_xlim = 5.5 
        train_xs = random.uniform(x_key, shape = (train_points, 3), minval = -train_xlim, maxval = train_xlim)
        train_ys = target_fn(train_xs).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = random.uniform(x_key, shape = (test_points, 3), minval = -test_xlim, maxval = test_xlim)
        test_xs = np.reshape(test_xs, (test_points, 3))
        test_ys = target_fn(test_xs)

    elif dname == 'Ishigami':
        train_points = 2000
        test_points = 5000
        noise_scale = 1e-1
        target_fn = lambda x: np.sin(x[:,0]) + 7*(np.sin(x[:,1])**2) + 0.1*(x[:,2]**4)*np.sin(x[:,0])

        train_xs = random.uniform(x_key, shape = (train_points, 3), minval = -np.pi/2, maxval = np.pi/2)
        train_ys = target_fn(train_xs).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = random.uniform(x_key, shape = (test_points, 3), minval = -2*np.pi/3, maxval = 2*np.pi/3)
        test_xs = np.reshape(test_xs, (test_points, 3))
        test_ys = target_fn(test_xs)
  
    elif dname == 'Environmental model':
        train_points = 200
        test_points = 500
        noise_scale = 1e-1

        M_tr = random.uniform(x_key, shape = (train_points, 1), minval = 7, maxval = 13)
        D_tr = random.uniform(x_key, shape = (train_points, 1), minval = 0.02, maxval = 0.12)
        L_tr = random.uniform(x_key, shape = (train_points, 1), minval = 0.01, maxval = 3)
        g_tr = random.uniform(x_key, shape = (train_points, 1), minval = 30.01, maxval = 30.295)
        train_xs = np.hstack((M_tr, D_tr, L_tr, g_tr))
        train_ys = env_model(train_xs[:,0], train_xs[:,1], train_xs[:,2], train_xs[:,3]).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        M_t = random.uniform(x_key, shape = (test_points, 1), minval = 5, maxval = 15)
        D_t = random.uniform(x_key, shape = (test_points, 1), minval = 0, maxval = 0.15)
        L_t = random.uniform(x_key, shape = (test_points, 1), minval = 0.01, maxval = 3.2)
        g_t = random.uniform(x_key, shape = (test_points, 1), minval = 23.71, maxval = 31)

        test_xs = np.hstack((M_t, D_t, L_t, g_t))
        test_xs = np.reshape(test_xs, (test_points, 4))
        test_ys = env_model(test_xs[:,0], test_xs[:,1], test_xs[:,2], test_xs[:,3])

    elif dname == 'Griewank':
        train_points = 200
        test_points = 500
        noise_scale = 1e-1

        train_xs = random.uniform(x_key, shape = (train_points, 4), minval = -500, maxval = 500)
        train_ys = griewank(train_xs).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = random.uniform(x_key, shape = (test_points, 4), minval = -600, maxval = 600)
        test_xs = np.reshape(test_xs, (test_points, 4))
        test_ys = griewank(test_xs)
  
    elif dname == 'Roos & Arnold':
        train_points = 200
        test_points = 500
        noise_scale = 1e-1
        d = 5

        train_xs = random.uniform(x_key, shape = (train_points, d), minval = 0, maxval = 0.8)
        train_ys = roos_arnold(train_xs).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = random.uniform(x_key, shape = (test_points, d), minval = 0, maxval = 1)
        test_xs = np.reshape(test_xs, (test_points, d))
        test_ys = roos_arnold(test_xs)

    elif dname == 'Friedman':
        train_points = 200
        test_points = 500
        noise_scale = 1e-1
        target_fn = lambda x: 10*np.sin(np.pi*x[:,0]*x[:,1]) + 20*((x[:,2] - 0.5)**2) + 10*x[:,3] + 5*x[:,4]

        train_xs = random.uniform(x_key, shape = (train_points, 5), minval = 0, maxval = 0.5)
        train_ys = target_fn(train_xs).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = random.uniform(x_key, shape = (test_points, 5), minval = 0, maxval = 1)
        test_xs = np.reshape(test_xs, (test_points, 5))
        test_ys = target_fn(test_xs)
        
    elif dname == 'Planar arm torque':
        train_points = 200
        test_points = 500
        noise_scale = 1e-1

        q_ddot = random.uniform(x_key,shape = (train_points, 2),minval = -np.pi, maxval = np.pi)
        q_dot = random.uniform(x_key,shape = (train_points, 2),minval = -np.pi, maxval = np.pi)
        q = random.uniform(x_key,shape = (train_points, 2),minval = -np.pi/2, maxval = np.pi/2)
        train_xs = np.hstack((q, q_dot, q_ddot))
        train_ys = planar_arm_torque(train_xs).reshape(len(train_xs),2)
        train_ys += noise_scale * random.normal(y_key, (train_points, 2))
        train_ys = train_ys[:,0].reshape(train_points, 1)  # approximating a first motor torch

        q_ddot = random.uniform(x_key,shape = (test_points, 2),minval = -2*np.pi, maxval = 2*np.pi)
        q_dot = random.uniform(x_key,shape = (test_points, 2),minval = -2*np.pi, maxval = 2*np.pi)
        q = random.uniform(x_key,shape = (test_points, 2),minval = -np.pi, maxval = np.pi)
        test_xs = np.hstack((q, q_dot, q_ddot))
        test_xs = np.reshape(test_xs, (test_points, 6))
        test_ys = planar_arm_torque(test_xs)
        test_ys = test_ys[:,0].reshape(test_points, 1)   # approximating a first motor torch

    elif dname == 'Sum of powers':
        train_points = 200
        test_points = 500
        noise_scale = 1e-1

        train_xs = random.uniform(x_key, shape = (train_points, 6), minval = -0.75, maxval = 0.75)
        train_ys = sum_of_powers(train_xs).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = random.uniform(x_key, shape = (test_points, 6), minval = -1, maxval = 1)
        test_xs = np.reshape(test_xs, (test_points, 6))
        test_ys = sum_of_powers(test_xs)
  
    elif dname == 'Ackley':
        train_points = 400
        test_points = 1000
        noise_scale = 1e-1
        d = 7

        train_xs = random.uniform(x_key, shape = (train_points, d), minval = -30, maxval = 30)
        train_ys = ackley(train_xs).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = random.uniform(x_key, shape = (test_points, d), minval = -32.768, maxval = 32.768)
        test_xs = np.reshape(test_xs, (test_points, d))
        test_ys = ackley(test_xs)

    elif dname == 'Piston simulation':
        train_points = 200
        test_points = 500
        noise_scale = 1e-1

        train_M = random.uniform(x_key, shape = (train_points, 1), minval = 30, maxval = 60)
        train_S = random.uniform(x_key, shape = (train_points, 1), minval = 0.005, maxval = 0.020)
        train_V_0 = random.uniform(x_key, shape = (train_points, 1), minval = 0.002, maxval = 0.010)
        train_k = random.uniform(x_key, shape = (train_points, 1), minval = 1000, maxval = 5000)
        train_P_0 = random.uniform(x_key, shape = (train_points, 1), minval = 90000, maxval = 110000)
        train_T_a = random.uniform(x_key, shape = (train_points, 1), minval = 290, maxval = 296)
        train_T_0 = random.uniform(x_key, shape = (train_points, 1), minval = 340, maxval = 360)
        train_xs = np.hstack((train_M, train_S, train_V_0, train_k, train_P_0, train_T_a, train_T_0))
        train_ys = piston(train_xs).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_M = random.uniform(x_key, shape = (test_points, 1), minval = 0, maxval = 90)
        test_S = random.uniform(x_key, shape = (test_points, 1), minval = 0.005, maxval = 0.03)
        test_V_0 = random.uniform(x_key, shape = (test_points, 1), minval = 0.00, maxval = 0.015)
        test_k = random.uniform(x_key, shape = (test_points, 1), minval = 10, maxval = 6000)
        test_P_0 = random.uniform(x_key, shape = (test_points, 1), minval = 80000, maxval = 120000)
        test_T_a = random.uniform(x_key, shape = (test_points, 1), minval = 285, maxval = 300)
        test_T_0 = random.uniform(x_key, shape = (test_points, 1), minval = 300, maxval = 400)
        test_xs = np.hstack((test_M, test_S, test_V_0, test_k, test_P_0, test_T_a, test_T_0))
        test_xs = np.reshape(test_xs, (test_points, 7))
        test_ys = piston(test_xs)

    elif dname == 'Robot arm':
        train_points = 200
        test_points = 500
        noise_scale = 1e-1

        train_xs_f = random.uniform(x_key, shape = (train_points, 4), minval = 0, maxval = np.pi)
        train_xs_s = random.uniform(x_key, shape = (train_points, 4), minval = 0, maxval = 0.5)
        train_xs = np.hstack((train_xs_f, train_xs_s))
        train_ys = robot_arm(train_xs).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs_f = random.uniform(x_key, shape = (test_points, 4), minval = 0, maxval = 2*np.pi)
        test_xs_s = random.uniform(x_key, shape = (test_points, 4), minval = 0, maxval = 1)
        test_xs = np.hstack((test_xs_f, test_xs_s))
        test_xs = np.reshape(test_xs, (test_points, 8))
        test_ys = robot_arm(test_xs)
  
    elif dname == 'Borehole':
        train_points = 2000
        test_points = 5000
        noise_scale = 1e-1

        rw_tr = random.uniform(x_key, shape = (train_points, 1), minval = 0.05, maxval = 0.15)
        r_tr = random.uniform(x_key, shape = (train_points, 1), minval = 100, maxval = 50000)
        Tu_tr = random.uniform(x_key, shape = (train_points, 1), minval = 63070, maxval = 115600)
        Hu_tr = random.uniform(x_key, shape = (train_points, 1), minval = 990, maxval = 1110)
        Tl_tr = random.uniform(x_key, shape = (train_points, 1), minval = 63.1, maxval = 116)
        Hl_tr = random.uniform(x_key, shape = (train_points, 1), minval = 700, maxval = 820)
        L_tr = random.uniform(x_key, shape = (train_points, 1), minval = 1120, maxval = 1680)
        Kw_tr = random.uniform(x_key, shape = (train_points, 1), minval = 9855, maxval = 12045)
        train_xs = np.hstack((rw_tr, r_tr, Tu_tr, Hu_tr, Tl_tr, Hl_tr, L_tr, Kw_tr))
        train_ys = borehole(rw_tr, r_tr, Tu_tr, Hu_tr, Tl_tr, Hl_tr, L_tr, Kw_tr).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        rw_t = random.uniform(x_key, shape = (test_points, 1), minval = 0.01, maxval = 0.2)
        r_t = random.uniform(x_key, shape = (test_points, 1), minval = 90, maxval = 50010)
        Tu_t = random.uniform(x_key, shape = (test_points, 1), minval = 63020, maxval = 115650)
        Hu_t = random.uniform(x_key, shape = (test_points, 1), minval = 950, maxval = 1150)
        Tl_t = random.uniform(x_key, shape = (test_points, 1), minval = 60, maxval = 120)
        Hl_t = random.uniform(x_key, shape = (test_points, 1), minval = 650, maxval = 900)
        L_t = random.uniform(x_key, shape = (test_points, 1), minval = 1100, maxval = 1700)
        Kw_t = random.uniform(x_key, shape = (test_points, 1), minval = 9800, maxval = 12100)
        test_xs = np.hstack((rw_t, r_t, Tu_t, Hu_t, Tl_t, Hl_t, L_t, Kw_t))
        test_xs = np.reshape(test_xs, (test_points, 8))
        test_ys = borehole(rw_t, r_t, Tu_t, Hu_t, Tl_t, Hl_t, L_t, Kw_t)
  
    elif dname == 'Styblinski-Tang':
        train_points = 400
        test_points = 1000
        noise_scale = 1e-1
        d = 9

        train_xs = random.uniform(x_key, shape = (train_points, d), minval = -5, maxval = 5)
        train_ys = styblinski_tang(train_xs).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = random.uniform(x_key, shape = (test_points, d), minval = -6, maxval = 6)
        test_xs = np.reshape(test_xs, (test_points, d))
        test_ys = styblinski_tang(test_xs)
  
    elif dname == 'PUMA560':
        train_points = 3693
        test_points = 4499
        noise_scale = 4e-1

        train_xs_f = numpy.loadtxt(os.getcwd()+'/init/data/puma8NH.test', usecols=range(0,9))[:,0:-1]
        torque3 = random.uniform(x_key, shape = (train_xs_f.shape[0], 1), minval = -0.5*1.2, maxval = 0.5*1.2)
        train_xs = np.hstack((train_xs_f, torque3))
        train_ys = numpy.loadtxt(os.getcwd()+'/init/data/puma8NH.test', usecols=range(0,9))[:,-1].reshape(len(train_xs),1)

        test_xs_f = numpy.loadtxt(os.getcwd()+'/init/data/puma8NH.data', usecols=range(0,9))[:,0:-1]
        torque3_t = random.uniform(x_key, shape = (test_xs_f.shape[0], 1), minval = -0.5*1.2, maxval = 0.5*1.2)
        test_xs = np.hstack((test_xs_f, torque3_t))
        test_ys = numpy.loadtxt(os.getcwd()+'/init/data/puma8NH.data', usecols=range(0,9))[:,-1]
  
    elif dname == 'Adapted Welch':
        train_points = 200
        test_points = 500
        noise_scale = 1e-1

        train_xlim = 0.5  #maxval of train
        test_xlim = 1 #maxval of test

        train_xs = random.uniform(x_key, shape = (train_points, 10), minval = -train_xlim, maxval = train_xlim)
        train_ys = adapt_welch(train_xs[:,0], train_xs[:,1], train_xs[:,2], train_xs[:,3], train_xs[:,4], train_xs[:,5], train_xs[:,6], train_xs[:,7], train_xs[:,8], train_xs[:,9]).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        test_xs = random.uniform(x_key, shape = (test_points, 10), minval = -test_xlim, maxval = test_xlim)
        test_xs = np.reshape(test_xs, (test_points, 10))
        test_ys = adapt_welch(test_xs[:,0], test_xs[:,1], test_xs[:,2], test_xs[:,3], test_xs[:,4], test_xs[:,5], test_xs[:,6], test_xs[:,7], test_xs[:,8], test_xs[:,9])
   
    elif dname == 'Wing weight':
        train_points = 2000
        test_points = 5000
        noise_scale = 1e-1

        S_w_tr = random.uniform(x_key, shape = (train_points, 1), minval = 150, maxval = 200)
        W_fw_tr = random.uniform(x_key, shape = (train_points, 1), minval = 220, maxval = 300)
        A_tr = random.uniform(x_key, shape = (train_points, 1), minval = 6, maxval = 10)
        L_tr = random.uniform(x_key, shape = (train_points, 1), minval = -10, maxval = 10)
        q_tr = random.uniform(x_key, shape = (train_points, 1), minval = 16, maxval = 45)
        lambda1_tr = random.uniform(x_key, shape = (train_points, 1), minval = 0.5, maxval = 1)
        t_c_tr = random.uniform(x_key, shape = (train_points, 1), minval = 0.08, maxval = 0.18)
        N_z_tr = random.uniform(x_key, shape = (train_points, 1), minval = 2.5, maxval = 6)
        W_dg_tr = random.uniform(x_key, shape = (train_points, 1), minval = 1700, maxval = 2500)
        W_p_tr = random.uniform(x_key, shape = (train_points, 1), minval = 0.025, maxval = 0.08)
        train_xs = np.hstack((S_w_tr, W_fw_tr, A_tr, L_tr, q_tr, lambda1_tr, t_c_tr, N_z_tr, W_dg_tr, W_p_tr))
        train_ys = wing_weight(S_w_tr, W_fw_tr, A_tr, L_tr, q_tr, lambda1_tr, t_c_tr, N_z_tr, W_dg_tr, W_p_tr).reshape(len(train_xs),1)
        train_ys += noise_scale * random.normal(y_key, (train_points, 1))

        S_w_t = random.uniform(x_key, shape = (test_points, 1), minval = 100, maxval = 250)
        W_fw_t = random.uniform(x_key, shape = (test_points, 1), minval = 200, maxval = 320)
        A_t = random.uniform(x_key, shape = (test_points, 1), minval = 0, maxval = 15)
        L_t = random.uniform(x_key, shape = (test_points, 1), minval = -20, maxval = 20)
        q_t = random.uniform(x_key, shape = (test_points, 1), minval = 0, maxval = 60)
        lambda1_t = random.uniform(x_key, shape = (test_points, 1), minval = 0, maxval = 1.5)
        t_c_t = random.uniform(x_key, shape = (test_points, 1), minval = 0.05, maxval = 0.25)
        N_z_t = random.uniform(x_key, shape = (test_points, 1), minval = 0.5, maxval = 8)
        W_dg_t = random.uniform(x_key, shape = (test_points, 1), minval = 1000, maxval = 3000)
        W_p_t = random.uniform(x_key, shape = (test_points, 1), minval = 0, maxval = 0.1)
        test_xs = np.hstack((S_w_t, W_fw_t, A_t, L_t, q_t, lambda1_t, t_c_t, N_z_t, W_dg_t, W_p_t))
        test_xs = np.reshape(test_xs, (test_points, 10))
        test_ys = wing_weight(S_w_t, W_fw_t, A_t, L_t, q_t, lambda1_t, t_c_t, N_z_t, W_dg_t, W_p_t)
  
    elif dname == 'Boston housing':
        noise_scale = 1e-1
        house_prices_dataset = datasets.load_boston()
        house_prices_df = pd.DataFrame(house_prices_dataset['data'])
        house_prices_df.columns = house_prices_dataset['feature_names']

        all_features = ['RM']
        X = house_prices_df[all_features].values
        y = house_prices_dataset['target']
        train_xs, test_xs, train_ys, test_ys = train_test_split(X, y, test_size=0.3, random_state=0)
        scaler_y = StandardScaler()
        train_ys = scaler_y.fit_transform(train_ys.reshape(-1, 1))
        scaler_X = StandardScaler()
        train_xs = scaler_X.fit_transform(train_xs)
        test_xs = scaler_X.transform(test_xs)
        train_ys = train_ys.reshape(-1, 1)
        test_ys = test_ys.reshape(-1, 1)

        train = Data(inputs = train_xs, targets = train_ys)
        test = Data(inputs = test_xs, targets = test_ys)
        return train, test, len(train_xs), len(test_xs), noise_scale, scaler_y, scaler_X
    
      
    elif dname == 'Abalone':
        data = numpy.loadtxt(os.getcwd()+'/init/data/abalone.data', usecols=[1,2,3,4,5,6,7,8])
        noise_scale = 1e-1
        X = data[:,0:-3]
        y = data[:,-1]
        train_xs = X[0:1880,:]
        train_ys = y[0:1880,].reshape(-1,1)

        test_xs = X[1880:,:]
        test_ys = y[1880:,].reshape(-1,1)
  
    elif dname == 'Naval propulsion plant':
        data = numpy.loadtxt(os.getcwd()+'/init/data/naval.txt', usecols=[2,3,7,13,17])
        noise_scale = 1e-1
        X = data[:,0:-1]
        y = data[:,-1]
        train_xs = X[0:5370,:]
        train_ys = y[0:5370,].reshape(-1,1)

        test_xs = X[5370:,:]
        test_ys = y[5370:,].reshape(-1,1)
  
    elif dname == 'Forest fire':
        data = pd.read_csv(os.getcwd()+'/init/data/forestfires.csv')
        noise_scale = 1e-1
        X = np.array(data[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH']])
        y = np.log((np.array(data['area']) + 1).astype(float))

        train_xs = X[0:200,:]
        train_ys = y[0:200:,].reshape(-1,1)

        test_xs = X[200:,:]
        test_ys = y[200:,].reshape(-1,1)
  
    elif dname == 'Parkinson':
        noise_scale = 1e-1
        data = pd.read_csv(os.getcwd()+'/init/data/parkinsons_updrs.data')
        X = np.array(data[['NHR', 'HNR', 'DFA', 'PPE', 'RPDE']])
        y = np.array(data['total_UPDRS']).astype(float)

        train_xs = X[0:2643,:]
        train_ys = y[0:2643:,].reshape(-1,1)

        test_xs = X[2643:,:]
        test_ys = y[2643:,].reshape(-1,1)
  
    else: 
        print('{} not implemented'.format(dname))
        return None, None, None, None, None, None, None
      

    train = Data(inputs = train_xs, targets = train_ys)
    test = Data(inputs = test_xs, targets = test_ys)

    return train, test, len(train_xs), len(test_xs), noise_scale, None, None


train, test, train_points, test_points, noise_scale, scaler_y, scaler_X  = generate_data(dname)