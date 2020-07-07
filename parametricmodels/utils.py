
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from scipy.stats import chi2
import seaborn as sns
plt.style.use('seaborn')


def tf_ravel_dict(tensor_dict):
    """
    output one dimensional tensor
    """
    return tf.concat([tf.reshape(w, [-1]) for w in tensor_dict.values()], axis = -1)


def params_ravel_to_dict(params_ravel, signature_params):
    shapes = [w.shape for w in signature_params.values()] # product of dimentions

    slice_indices = [np.prod(sh, dtype = np.int32) for sh in shapes]
    slice_indices.insert(0, 0)
    slice_indices = np.cumsum(slice_indices)

    params_list = [tf.reshape(params_ravel[i:j], sh) for i, j, sh in zip(slice_indices[:-1], slice_indices[1:], shapes)]
    return {key: w for key, w in zip(signature_params.keys(), params_list)}

# TODO: better: use the signature of functional equation
def ravel_inputs(functional_equation, signature_params):

    def functional_equation_ravel(input_labels, params_ravel):
        params = params_ravel_to_dict(params_ravel, signature_params)
        return functional_equation(input_labels, params)

    return functional_equation_ravel


def ravel_loss(loss, signature_params, loss_from_features = False):
    '''
    retrurn a funciton that accepts flat parameter inputs
    '''
    if loss_from_features:
        def loss_ravel_equation(input_labels, predictions, input_features, params_ravel):
            params = params_ravel_to_dict(params_ravel, signature_params)
            return loss(input_labels, predictions, input_features, params)
    else:
        def loss_ravel_equation(input_labels, predictions, params_ravel):
            params = params_ravel_to_dict(params_ravel, signature_params)
            return loss(input_labels, predictions, params)

    return loss_ravel_equation


def tensor_to_numpy_dict(tensor_dict):
    return {key:val.numpy() for key,val in tensor_dict.items()}

def numpy_ravel(arr):
    """
    returns one dimensional numpy array
    """
    if not isinstance(arr, (np.ndarray, np.number)):
        arr = arr.numpy()
    return np.array(arr)[np.newaxis] if np.ndim(arr) == 0 else arr.ravel()


def ravel_dicts(list_of_dicts):
    if not list_of_dicts:
        return np.array([])

    num_dicts = len(list_of_dicts)
    first = np.concatenate([numpy_ravel(val) for val in list_of_dicts[0].values()])

    res = np.empty(shape = (len(list_of_dicts), *first.shape), dtype = first.dtype)
    res[0,:] = first
    for i, d in enumerate(list_of_dicts):
        res[i, :] = np.concatenate([numpy_ravel(val) for val in d.values()])
    return res




def quadratic_form(m, V):
    def quad(z):
        z0 = z - m.reshape(1, -1)
        V_inv = np.linalg.inv(V)

        return np.sum(z0*np.matmul(z0, V_inv), axis = 1)
    return quad


def plot_confidence_2d_projection(m, cov, proj, res_ravel = None, quantiles = [0.95, 0.975, 0.99, 0.9995], figsize = (10,10)):
    '''
    `m`     np.array, region center, e.g. parameter estimator
    `cov`   np.array, full covariance matrix of size = (n_cov, n_cov), e.g. from parameter estimation
    `proj`  np.array, projection matrix on 2-plane of size = (n_cov, 2)
    `res_ravel` np.array, if given -- ovelayed with scatter plots

    Plots contours of confidence regions for a given 2d projection of a multivariate normal distribution
    '''
    contour_levels = chi2(df = 2).ppf(quantiles)
    m_proj = proj.T @ m
    cov_proj = proj.T @ cov @ proj
    quad = quadratic_form(m_proj, cov_proj)

    lim_val = np.linalg.eig(cov_proj)[0].max()**.5*contour_levels.max()**.5

    ux = np.linspace(m_proj[0] - lim_val, m_proj[0] + lim_val, 100)
    uy = np.linspace(m_proj[1] - lim_val, m_proj[1] + lim_val, 100)
    xs, ys = np.meshgrid(ux, uy)
    xy = np.concatenate([xs.reshape(-1,1), ys.reshape(-1,1)], axis = -1)

    fig, ax = plt.subplots(figsize = figsize)
    ax.contour(xs, ys, quad(xy).reshape(100,100), contour_levels, colors  = 'red')

    if res_ravel is not None:
        res_proj = res_ravel @ proj
        ax.scatter(res_proj[:,0],res_proj[:,1], alpha = 0.5)

    return quad


@tf.function
def sq_loss(labels, predictions, params = None):
    return tf.square(labels - predictions)
