from init.utils import *
from init.datasets import *

# Put the data into the correct format
X_train = numpy.array(train.inputs)
y_train = numpy.array(train.targets)

X_val = numpy.array(test.inputs)
y_val = numpy.array(test.targets)

class NN():
    def __init__(self, x_dim, y_dim, hidden_size, init_stddev_1_w, init_stddev_1_b, 
                 init_stddev_2_w, n, learning_rate, ens):

        # setting up as for a usual NN
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size 
        self.n = n
        self.learning_rate = learning_rate
        
        # set up NN
        self.inputs = tf.placeholder(tf.float64, [None, x_dim], name='inputs')
        self.y_target = tf.placeholder(tf.float64, [None, y_dim], name='target')
        activation_fns = [tensorflow.keras.activations.selu, tf.nn.tanh, tensorflow.keras.activations.gelu, tensorflow.keras.activations.softsign, tf.math.erf, tf.nn.swish, tensorflow.keras.activations.linear]
        
        if ens <= len(activation_fns)-1:
            self.layer_1_w = tf.layers.Dense(hidden_size, activation = activation_fns[ens], kernel_initializer = tf.random_normal_initializer(mean=0., stddev = init_stddev_1_w), bias_initializer = tf.random_normal_initializer(mean=0., stddev=init_stddev_1_b))
            self.layer_2_w = tf.layers.Dense(hidden_size, activation = activation_fns[ens], kernel_initializer = tf.random_normal_initializer(mean=0., stddev = init_stddev_2_w), bias_initializer = tf.random_normal_initializer(mean=0., stddev=init_stddev_2_w))
        
        else:
            af_ind = randint(0,len(activation_fns)-1)
            self.layer_1_w = tf.layers.Dense(hidden_size, activation = activation_fns[af_ind], kernel_initializer = tf.random_normal_initializer(mean=0., stddev = init_stddev_1_w), bias_initializer = tf.random_normal_initializer(mean=0., stddev=init_stddev_1_b)) 
            self.layer_2_w = tf.layers.Dense(hidden_size, activation = activation_fns[af_ind], kernel_initializer = tf.random_normal_initializer(mean=0., stddev = init_stddev_2_w), bias_initializer = tf.random_normal_initializer(mean=0., stddev=init_stddev_2_w))
        
        self.layer_1 = self.layer_1_w.apply(self.inputs)
        self.layer_2 = self.layer_2_w.apply(self.layer_1)
        self.output_w = tf.layers.Dense(y_dim, activation=None, use_bias=False, kernel_initializer = tf.random_normal_initializer(mean=0., stddev=init_stddev_2_w))
        self.output = self.output_w.apply(self.layer_2)
        
        # set up loss and optimiser - this is modified later with anchoring regularisation
        self.opt_method = tf.train.AdamOptimizer(self.learning_rate)
        self.mse_ = 1/tf.shape(self.inputs, out_type=tf.int64)[0] * tf.reduce_sum(tf.square(self.y_target - self.output))
        self.loss_ = 1/tf.shape(self.inputs, out_type=tf.int64)[0] * tf.reduce_sum(tf.square(self.y_target - self.output))
        self.optimizer = self.opt_method.minimize(self.loss_)
        return
    
    
    def get_weights(self, sess):
        '''method to return current params'''
        
        ops = [self.layer_1_w.kernel, self.layer_1_w.bias, self.layer_2_w.kernel, self.layer_2_w.bias, self.output_w.kernel]
        w1, b1, w2, b2, w3 = sess.run(ops) 
        
        return w1, b1, w2, b2, w3
    
    
    def anchor(self, sess, lambda_):   #lambda_anchor
        '''regularise around initial parameters''' 
        
        w1, b1, w2, b2, w3 = self.get_weights(sess)
        self.w1_init, self.b1_init, self.w2_init, self.b2_init, self.w3_init = w1, b1, w2, b2, w3
        
        loss = lambda_[0]*tf.reduce_sum(tf.square(self.w1_init - self.layer_1_w.kernel))
        loss += lambda_[1]*tf.reduce_sum(tf.square(self.b1_init - self.layer_1_w.bias))
        loss += lambda_[2]*tf.reduce_sum(tf.square(self.w2_init - self.layer_2_w.kernel))
        loss += lambda_[2]*tf.reduce_sum(tf.square(self.b2_init - self.layer_2_w.bias))
        loss += lambda_[2]*tf.reduce_sum(tf.square(self.w3_init - self.output_w.kernel))

        # combine with original loss
        self.loss_ = self.loss_ + 1/tf.shape(self.inputs, out_type=tf.int64)[0] * loss 
        self.optimizer = self.opt_method.minimize(self.loss_)
        return
      
    def predict(self, x, sess):
        '''predict method'''
        
        feed = {self.inputs: x}
        y_pred = sess.run(self.output, feed_dict=feed)
        return y_pred
    
# hyperparameters
n = X_train.shape[0]
x_dim = X_train.shape[1]
y_dim = y_train.shape[1]
n_ensembles = 5
hidden_size = 128
init_stddev_1_w =  np.sqrt(10)
init_stddev_1_b = init_stddev_1_w # set these equal
init_stddev_2_w = 1.0/np.sqrt(hidden_size) # normal scaling
data_noise = 0.01 #estimated noise variance, feel free to experiment with different values
lambda_anchor = data_noise/(np.array([init_stddev_1_w, init_stddev_1_b, init_stddev_1_w, init_stddev_1_b, init_stddev_1_w, init_stddev_1_b, init_stddev_1_w, init_stddev_1_b, init_stddev_1_w, init_stddev_1_b, init_stddev_2_w])**2)
n_epochs = 1000
learning_rate = 0.01


NNs=[]
y_prior=[]
tf.reset_default_graph()
sess = tf.Session()

# loop to initialise all ensemble members, get priors
for ens in range(0,n_ensembles):
    NNs.append(NN(x_dim, y_dim, hidden_size, 
                  init_stddev_1_w, init_stddev_1_b, init_stddev_2_w, n, learning_rate, ens))
    
    # initialise only unitialized variables - stops overwriting ensembles already created
    global_vars = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))
    
    # do regularisation now that we've created initialisations
    NNs[ens].anchor(sess, lambda_anchor)  #Do that if you want to minimize the anchored loss
    
    # save their priors
    y_prior.append(NNs[ens].predict(X_val, sess))

for ens in range(0,n_ensembles):
    
    feed_b = {}
    feed_b[NNs[ens].inputs] = X_train
    feed_b[NNs[ens].y_target] = y_train
    print('\nNN:',ens)
    
    ep_ = 0
    while ep_ < n_epochs:    
        ep_ += 1
        blank = sess.run(NNs[ens].optimizer, feed_dict=feed_b)
        if ep_ % (n_epochs/5) == 0:
            loss_mse = sess.run(NNs[ens].mse_, feed_dict=feed_b)
            loss_anch = sess.run(NNs[ens].loss_, feed_dict=feed_b)
            print('epoch:', ep_, ', mse_', np.round(loss_mse*1e3,3), ', loss_anch', np.round(loss_anch*1e3,3))
            # the anchored loss is minimized, but it's useful to keep an eye on mse too

# run predictions
y_pred=[]
for ens in range(0,n_ensembles):
    y_pred.append(NNs[ens].predict(X_val, sess))

"""Report metrics:"""

method_means = np.mean(np.array(y_pred)[:,:,0], axis=0).reshape(-1,)
method_stds = np.sqrt(np.square(np.std(np.array(y_pred)[:,:,0],axis=0, ddof=1)) + data_noise).reshape(-1,)
report_res(dname, y_val, "RAFs Ensemble", method_means, method_stds)