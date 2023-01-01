# import modules
from init.utils import * 
from init.datasets import *
from init.gp import *

########## Bootrapped Ensemble of NNs Coupled with Randomized Prior Functions (Osband et al.) ##########

# put data in the correct format
X = numpy.array(train.inputs)
y = numpy.array(train.targets)

x_grid = numpy.array(test.inputs)
y_val = numpy.array(test.targets)

def get_randomized_prior_nn():

    # shared input of the network
    net_input = Input(shape=(X.shape[1],), name='input')

    # trainable network body
    trainable_net = Sequential([Dense(16, 'tanh'), 
                                Dense(16, 'tanh')], 
                               name='trainable_net')(net_input)
    
    # trainable network output
    trainable_output = Dense(1, 'linear', name='trainable_out')(trainable_net)

    # prior network body - we use trainable=False to keep the network output random 
    prior_net = Sequential([Dense(16, 'tanh', kernel_initializer='glorot_normal',trainable=False),  
                            Dense(16, 'tanh', kernel_initializer='glorot_normal',trainable=False)],     
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
base_model = KerasRegressor(build_fn=get_randomized_prior_nn, 
                            epochs=1000, batch_size=10, verbose=0)

# create a bagged ensemble of 5 base models
bag = BaggingRegressor(base_estimator=base_model, n_estimators=5, verbose=2)

bag.fit(X, y.ravel())

# individual predictions on the grid of values
y_grid = numpy.array([e.predict(x_grid.reshape(-1,X.shape[1])) for e in bag.estimators_]).T
trainable_grid = numpy.array([Model(inputs=e.model.input,outputs=e.model.get_layer('trainable_out').output).predict(x_grid.reshape(-1,X.shape[1])) for e in bag.estimators_]).T
prior_grid = numpy.array([Model(inputs=e.model.input,outputs=e.model.get_layer('prior_scale').output).predict(x_grid.reshape(-1,X.shape[1])) for e in bag.estimators_]).T

# display results

method_means = np.array(y_grid).mean(axis=1).reshape(-1,)
method_stds = np.array(y_grid).std(axis=1).reshape(-1,)
viz_one_d(dname, train, test, "RP-param", method_means, method_stds, predictions, False) # change to True if you want to save the plot

# report metrics

report_res(dname, test, y_val, "RP-param", method_means, method_stds, scaler_X, scaler_y)