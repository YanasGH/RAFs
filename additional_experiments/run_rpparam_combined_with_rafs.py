import os
import pandas as pd
from collections import namedtuple
from random import randint
from init.utils import *

"""## Bootrapped Ensemble of NNs Coupled with Randomized Prior Functions (Osband et al.)"""

Data = namedtuple('Data', ['inputs', 'targets'])

#Parkinson's datset
noise_scale = 1e-1
data = pd.read_csv(os.getcwd()[0:-22] + 'main_experiments/init/data/parkinsons_updrs.data')
X = np.array(data[['NHR', 'HNR', 'DFA', 'PPE', 'RPDE']])
y = np.array(data['total_UPDRS']).astype(float)

train_xs = X[0:2643,:]
train_ys = y[0:2643:,].reshape(-1,1)
test_xs = X[2643:,:]
test_ys = y[2643:,].reshape(-1,1)

train = Data(inputs = train_xs, targets = train_ys)
test = Data(inputs = test_xs, targets = test_ys)


#Put data in the correct format
X = numpy.array(train.inputs)
y = numpy.array(train.targets)

x_grid = numpy.array(test.inputs)
y_val = numpy.array(test.targets)

print("----- Running RP-param combined with RAFs methodology on Parkinson's dataset: -----")

used_afs = []
def get_raf(used_afs, activation_fns):
    """ Get unique random activation functions (AFs) if cardinality of the 
    set of Afs is larger or equal to the number of ensemble members. """
    
    af_ind = randint(0,len(activation_fns)-1)
    if len(used_afs)<len(activation_fns):
        if af_ind in used_afs:
            get_raf(activation_fns)
        else:
            used_afs.append(af_ind)
            return af_ind
    else:
        return af_ind

def get_randomized_prior_nn():
    """ Bootrapped Ensemble of NNs Coupled with Randomized Prior Functions (RP-param) """

    # shared input of the network
    net_input = Input(shape=(X.shape[1],), name='input')
    
    activation_fns = [tensorflow.keras.activations.selu, tf.nn.tanh, tensorflow.keras.activations.gelu, tensorflow.keras.activations.softsign, tf.math.erf, tf.nn.swish, tensorflow.keras.activations.linear]
    af_ind = get_raf(used_afs, activation_fns)
    
    # trainable network body
    trainable_net = Sequential([Dense(16, activation_fns[af_ind]), 
                                Dense(16, activation_fns[af_ind])], 
                               name='trainable_net')(net_input)
    
    # trainable network output
    trainable_output = Dense(1, 'linear', name='trainable_out')(trainable_net)

    # prior network body - we use trainable=False to keep the network output random 
    prior_net = Sequential([Dense(16, activation_fns[af_ind], kernel_initializer='glorot_normal',trainable=False),  
                            Dense(16, activation_fns[af_ind], kernel_initializer='glorot_normal',trainable=False)], 
                           name='prior_net')(net_input)
    # prior network output
    prior_output = Dense(1, 'linear', kernel_initializer='glorot_normal', trainable=False, name='prior_out')(prior_net)
    
    # using a lambda layer so we can control the weight (beta) of the prior network
    prior_output = Lambda(lambda x: x * 3.0, name='prior_scale')(prior_output)

    # lastly, we use a add layer to add both networks together and get Q
    add_output = add([trainable_output, prior_output], name='add')

    # defining the model and compiling it
    model = Model(inputs=net_input, outputs=add_output)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    
    return model

# wrapping the base model around a sklearn estimator
base_model = KerasRegressor(build_fn=get_randomized_prior_nn, epochs=1000, batch_size=10, verbose=0)

# create a bagged ensemble of 5 base models
bag = BaggingRegressor(base_estimator=base_model, n_estimators=5, verbose=2)

bag.fit(X, y.ravel())

# individual predictions on the grid of values
y_grid = numpy.array([e.predict(x_grid.reshape(-1,X.shape[1])) for e in bag.estimators_]).T
trainable_grid = numpy.array([Model(inputs = e.model.input,outputs = e.model.get_layer('trainable_out').output).predict(x_grid.reshape(-1, X.shape[1])) for e in bag.estimators_]).T
prior_grid = numpy.array([Model(inputs = e.model.input,outputs = e.model.get_layer('prior_scale').output).predict(x_grid.reshape(-1, X.shape[1])) for e in bag.estimators_]).T


"""Report metrics:"""
method_means = np.array(y_grid).mean(axis=1).reshape(-1,)
method_stds = np.array(y_grid).std(axis=1).reshape(-1,)
report_res("Parkinsons", y_val, "RP-param", method_means, method_stds)