#import modules
from init.utils import * 
from init.datasets import *
from init.gp import *

########## Deep Ensemble (Lakshminarayanan et al.) ##########
### Code mainly based on the repository for the paper "Bayesian Deep Ensembles via the Neural Tangent Kernel" by He et al., which can be found at https://github.com/bobby-he/bayesian-ntk ###

from bayesian_ntk.models import homoscedastic_model
from bayesian_ntk.train_utils import fetch_new_predict_fn, fetch_regularisation_fn
from bayesian_ntk import config
from bayesian_ntk import train_utils

import jax.scipy as sp
from jax.example_libraries import optimizers

from jax.tree_util import tree_all
from jax.tree_util import tree_map

from neural_tangents._src.utils.utils import canonicalize_get
from neural_tangents._src.utils.utils import get_namedtuple

import collections

# random seed
key = random.PRNGKey(10)

# additional supporting functions for the NTKGP-param and Deep Ensemble: 
Gaussian = collections.namedtuple('Gaussian', 'mean standard_deviation')

def _get_matrices(kernel_fn, x_train, x_test, get, compute_cov):
    get = _get_dependency(get, compute_cov)
    kdd = kernel_fn(x_train, None, get)
    ktd = kernel_fn(x_test, x_train, get)
    if compute_cov:
        ktt = kernel_fn(x_test, x_test, get)
    else:
        ktt = None
    return kdd, ktd, ktt

# utility functions
def _get_dependency(get, compute_cov):
    """Figure out dependency for get."""
    _, get = canonicalize_get(get)
    for g in get:
        if g not in ['nngp', 'ntk']:
            raise NotImplementedError('Can only get either "nngp" or "ntk" predictions, got %s.' % g)
    get_dependency = ()
    if 'nngp' in get or ('ntk' in get and compute_cov):
        get_dependency += ('nngp',)
    if 'ntk' in get:
        get_dependency += ('ntk',)
    return get_dependency

def _add_diagonal_regularizer(covariance, diag_reg=0.):
    dimension = covariance.shape[0]
    return covariance + diag_reg * np.eye(dimension)

def _inv_operator(g_dd, diag_reg=0.0):
    g_dd_plus_reg = _add_diagonal_regularizer(g_dd, diag_reg)
    return lambda vec: sp.linalg.solve(g_dd_plus_reg, vec, assume_a='pos')


def _make_flatten_uflatten(g_td, y_train):
    """Create the flatten and unflatten utilities."""
    output_dimension = y_train.shape[-1]

    def fl(fx):
        """Flatten outputs."""
        return np.reshape(fx, (-1,))

    def ufl(fx):
        """Unflatten outputs."""
        return np.reshape(fx, (-1, output_dimension))

    if y_train.size > g_td.shape[-1]:
        out_dim, ragged = divmod(y_train.size, g_td.shape[-1])
        if ragged or out_dim != output_dimension:
            raise ValueError('The batch size of `y_train` must be the same as the'
                       ' last dimension of `g_td`')
        fl = lambda x: x
        ufl = lambda x: x
    return fl, ufl

def _mean_prediction(op, g_td, y_train):
    """Compute the mean prediction of a Gaussian process.
    Args:
    op: Some vector operator that projects the data along the relevant
      directions, op(vec, dt) = M^{-1} @ vec
    g_td: A kernel relating training data with test data. The kernel should be
      an `np.ndarray` of shape [n_test * output_dim, n_train * output_dim] or
      [n_test, n_train].
    y_train: An `np.ndarray` of shape [n_train, output_dim] of targets for the
      training data.
    Returns:
    The mean prediction of the GP. `g_td @ op @ y_train`.
    """
    fl, ufl = _make_flatten_uflatten(g_td, y_train)

    mean_pred = op(fl(y_train))
    mean_pred = np.dot(g_td, mean_pred)
    return ufl(mean_pred)

def _posterior_std(op, g_td, g_tt, output_noise_var):
    """Computes the test posterior standard deviation (with output noise) for nngp
     or ntkgp.
    """
    cov = op(np.transpose(g_td))
    pred_var = np.diag(g_tt - np.dot(g_td, cov)) + output_noise_var
    return np.sqrt(pred_var)

def _arr_is_on_cpu(x):
    # utility function from neural_tangents
    if hasattr(x, 'device_buffer'):
        return 'CPU' in str(x.device_buffer.device())

    if isinstance(x, np.ndarray):
        return True

    raise NotImplementedError(type(x))


def _is_on_cpu(x):
    # utility function from neural_tangents
    return tree_all(tree_map(_arr_is_on_cpu, x))

def gp_inference(kernel_fn, x_train, y_train, x_test, get, diag_reg=0., compute_cov=True):
    """Compute the mean and standard deviation of the `posterior` of NNGP & NTKGP.
    Args:
    kernel_fn: A kernel function that computes NNGP and NTK.
    x_train: A `np.ndarray`, representing the training data.
    y_train: A `np.ndarray`, representing the labels of the training data.
    x_test: A `np.ndarray`, representing the test data.
    get: string, the mode of the Gaussian process, either "nngp" or "ntk", or a
      tuple, or None. If `None` then both `nngp` and `ntk` predictions are
      returned.
    diag_reg: A float, representing the output noise variance.
    compute_cov: A boolean. If `True` computing both `mean` and `variance` and
      only `mean` otherwise.
    Returns:
    Either a Gaussian(`mean`, `standard deviation`) namedtuple or `mean` of the
    GP posterior.
    """
    if get is None:
        get = ('nngp', 'ntk')
    kdd, ktd, ktt = _get_matrices(kernel_fn, x_train, x_test, get, compute_cov)
    gp_inference_mat = (_gp_inference_mat_jit_cpu if _is_on_cpu(kdd) else
                      _gp_inference_mat_jit)
    return gp_inference_mat(kdd, ktd, ktt, y_train, get, diag_reg)

@get_namedtuple('Gaussians')
def _gp_inference_mat(kdd, ktd, ktt, y_train, get, diag_reg=0.):
    """Compute the mean and standard deviation of the `posterior` of NNGP & NTKGP.
    Args:
    kdd: A train-train `Kernel` namedtuple.
    ktd: A test-train `Kernel` namedtuple.
    ktt: A test-test `Kernel` namedtuple.
    y_train: A `np.ndarray`, representing the train targets.
    get: string, the mode of the Gaussian process, either "nngp" or "ntk", or a
      tuple, or `None`. If `None` then both `nngp` and `ntk` predictions are
      returned.
    diag_reg: A float, representing the strength of the regularization.
    Returns:
    Either a Gaussian(`mean`, `standard deviation`) namedtuple or `mean` of the
    GP posterior.
    """
    out = {}
    if get is None:
        get = ('nngp', 'ntk')
    if 'nngp' in get:
        op = _inv_operator(kdd.nngp, diag_reg)
        pred_mean = _mean_prediction(op, ktd.nngp, y_train)
        pred_mean = pred_mean.reshape(-1,)
        if ktt is not None:
            pred_std = _posterior_std(op, ktd.nngp, ktt.nngp, diag_reg)
        out['nngp'] = (
            Gaussian(pred_mean, pred_std) if ktt is not None else pred_mean)

        if 'ntk' in get:
            op = _inv_operator(kdd.ntk, diag_reg)
            pred_mean = _mean_prediction(op, ktd.ntk, y_train)
            pred_mean = pred_mean.reshape(-1,)
        if ktt is not None:
            pred_std = _posterior_std(op, ktd.ntk, ktt.ntk, diag_reg)
        out['ntk'] = (Gaussian(pred_mean, pred_std) if ktt is not None else pred_mean)

    return out

_gp_inference_mat_jit = jit(_gp_inference_mat, static_argnums=(4,))

_gp_inference_mat_jit_cpu = jit(_gp_inference_mat, static_argnums=(4,),
                                backend='cpu')

# defining an NN
init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(512, W_std=1.5, b_std=0.05), stax.Erf(),
    stax.Dense(512, W_std=1.5, b_std=0.05), stax.Erf(),
    stax.Dense(1, W_std=1.5, b_std=0.05)
)

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnums=(2,))

# extracting NTKGP and NNGP moments (Gaussian processes)
analytic_ntkgp_moments, analytic_nngp_moments = gp_inference(
    kernel_fn = kernel_fn,
    x_train = train.inputs,
    y_train = train.targets,
    x_test = test.inputs,
    get = ('ntk', 'nngp'),
    diag_reg = config.NOISE_SCALE**2,
    compute_cov = True
)

predictions = {
    'NTKGP analytic': analytic_ntkgp_moments,
    'NNGP analytic': analytic_nngp_moments
};

method_input_dct = {"Deep Ensemble": 'deep_ensemble'}

# train function
dim = train.inputs.reshape(len(train.inputs), -1).shape[1]
def train_model(
    key,
    train_method,
    train_data,
    test_data,
    activation,
    parameterization,
    learning_rate,
    training_steps,
    noise_scale,
    W_std,
    b_std,
    width,
    depth,
):
    """Train a single baselearner model and calculate test predictions.

    Args:
        key: jax.random.PRNGKey instance
        train_method (str): Ensemble method
        train_data: Tuple of training inputs and targets
        test_data: Tuple of test inputs and targets
        activation (str): Activation function
        parameterization (str): Parameterization
        learning_rate (float): Learning rate
        training_steps (int): Number of gradient updates
        noise_scale (float): output noise standard deviation
        W_std (float): Weight standard deviation
        b_std (float): Bias standard deviation
        width (int): Hidden layer width
        depth (int): Number of hidden layers

    Returns:
        Model predictions on `test_data`
    """

    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_update = jit(opt_update)

    # get model
    init_fn, predict_fn, _ = homoscedastic_model(
         W_std,
         b_std,
         width,
         depth,
         activation,
         parameterization
     )

    # initialise initial parameters
    _, init_params = init_fn(key, (-1, dim))

    # initialise auxiliary (non-trainable) parameters for JVPs in NTKGP methods
    # or extra forward pass in RP-fn method
    key, subkey = random.split(key)
    _, aux_params = init_fn(subkey, (-1, dim))

    # define `train_method` dependent modified forward pass and regularisation
    new_predict_fn = fetch_new_predict_fn(
        predict_fn,
        train_method,
        init_params,
        aux_params
    )
    new_predict_fn = jit(new_predict_fn)

    reg_fn = fetch_regularisation_fn(
        train_method,
        init_params,
        parameterization,
        W_std,
        b_std
    )
    def mse_loss(params, x, y):
        preds = new_predict_fn(params, x)
        return np.mean((preds - y) ** 2)

    train_size = len(train_data.inputs)
    reg_coef = noise_scale**2 / train_size

    @jit
    def loss(params, x, y):
        return 0.5 * mse_loss(params, x, y) + 0.5 * reg_coef * reg_fn(params)

    @jit
    def grad_loss(state, x, y):
        params = get_params(state)
        return grad(loss)(params, x, y)

    opt_state = opt_init(init_params)

    for i in range(training_steps):
        opt_state = opt_update(i, grad_loss(opt_state, *train_data), opt_state)

    final_params = get_params(opt_state)
    fx_final_test = new_predict_fn(final_params, test_data.inputs)

    return fx_final_test

# ensemble configuration
config.ENSEMBLE_SIZE = 5
train_config = config.get_train_config('default')
ensemble_key = random.split(key, config.ENSEMBLE_SIZE)
train_baselearner = lambda key, train_method: train_model(key, train_method, train, test, parameterization = 'standard', **train_config)
train_ensemble = lambda train_method: vmap(train_baselearner, (0, None))(ensemble_key, train_method)

# train ensembles
ensemble_methods_list = ['Deep Ensemble']

# this may take a few minutes
for method_idx, method in enumerate(ensemble_methods_list):
    method_input_str = method_input_dct[method]
    print(f"Training ensemble method {method_idx+1}/{len(ensemble_methods_list)}: {method}")
    baselearners_test_pred = train_ensemble(method_input_str)
    ensemble_mean = np.mean(baselearners_test_pred, axis = 0).reshape(-1,)
    ensemble_var = np.var(baselearners_test_pred, axis = 0, ddof = 1).reshape(-1,)
    ensemble_std = np.sqrt(ensemble_var + config.NOISE_SCALE ** 2)
    predictions.update(
        {
            method: Gaussian(ensemble_mean, ensemble_std)
        }
    )

# display the results
method_moments = predictions["Deep Ensemble"]
method_means = method_moments.mean.reshape(-1,)
method_stds = method_moments.standard_deviation.reshape(-1,)
viz_one_d(dname, train, test, "Deep Ensemble", method_means, method_stds, predictions, False) # change to True if you want to save the plot

# report metrics
for method in ensemble_methods_list:
    method_moments = predictions[method]
    method_means = method_moments.mean.reshape(-1,)
    method_stds = method_moments.standard_deviation.reshape(-1,)
    report_res(dname, test, test.targets, method, method_means, method_stds, scaler_X, scaler_y)
    print('')