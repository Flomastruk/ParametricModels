
import os
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

from collections import OrderedDict

from classes import ModelFromConcreteFunction
from utils import ravel_inputs, tf_ravel_dict,ravel_dicts


@tf.function
def linear_equation(input_features, params):
    return tf.squeeze(tf.matmul(input_features, params['W']))+ params['b']

@tf.function
def msq_loss(labels, predictions, params = None):
    return tf.reduce_mean(tf.square(labels - predictions))

@tf.function
def sq_loss(labels, predictions, params = None):
    return tf.square(labels - predictions)


def functional_gaussian_simulation_scheme(X_mean, X_cov, params, y_ssq, functional_equation):
    # @tf.function
    def concrete_functional_simulation_scheme(num_samples):
        X_sim = np.random.multivariate_normal(mean = X_mean, cov = X_cov, size = num_samples).astype(np.float32)
        err = np.random.normal(loc = 0., size = num_samples, scale = np.sqrt(y_ssq)).astype(np.float32)
        y_sim = functional_equation(X_sim, params) + err

        return tf.Variable(X_sim) , y_sim
    return concrete_functional_simulation_scheme

#@tf.function
def linear_gaussian_simulation_scheme(X_mean, X_cov, params, y_ssq):
    # @tf.function
    def concrete_linear_simulation_scheme(num_samples):
        X_sim = np.random.multivariate_normal(mean = X_mean, cov = X_cov, size = num_samples).astype(np.float32)
        err = np.random.normal(loc = 0., size = num_samples, scale = np.sqrt(y_ssq)).astype(np.float32)
        y_sim = linear_equation(X_sim, params) + err

        return tf.Variable(X_sim) , y_sim
    return concrete_linear_simulation_scheme


def quadratic_form(m, V):
    def quad(z):
        z0 = z - m.reshape(1, -1)
        V_inv = np.linalg.inv(V)

        return np.sum(z0*np.matmul(z0, V_inv), axis = 1)
    return quad


from scipy.stats import chi2

def plot_confidence_2d_projection(m, cov, proj, res_ravel = None, quantiles = [0.95, 0.975, 0.99, 0.9995], figsize = (10,10)):
    '''
    `m`     np.array, region center, e.g. parameter estimator
    `cov`   np.array, full covariance matrix of size = (n_cov, n_cov), e.g. from parameter estimation
    `proj`  np.array, projection matrix on 2-plane of size = (n_cov, 2)
    `res_ravel` np.array, if given -- ovelayed with scatter plots
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




## Part I. Ordinary Least Squares

W0 = tf.Variable([[-.5],[.8]], dtype = tf.float32)
b0 = tf.Variable(1.0, dtype = tf.float32)
linear_params0 = OrderedDict({'W': W0, 'b': b0})

mc = ModelFromConcreteFunction(linear_equation, model_name = "test_c_model", params = linear_params0)
mc.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
mc.loss = sq_loss

X_cov = np.array([[1,.95], [.95,1]], dtype = np.float32)
mc.simulation_scheme = linear_gaussian_simulation_scheme(
    np.array([0., 0.], dtype = np.float32),
    X_cov,
    linear_params0,
    y_ssq = 9
    )

tmp_x, tmp_y = mc.simulation_scheme(100)
mc.dldw2_plug_in_estimator(tmp_x, tmp_y, linear_params0, loss = sq_loss)
mc.d2ld2w_plug_in_estimator(tmp_x, tmp_y, linear_params0, loss = sq_loss)
cov_tmp = mc.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, linear_params0, loss = sq_loss)


mc.fit(
    tmp_x, tmp_y
    , num_steps = tf.Variable(2000)
    , loss = sq_loss
    , params = linear_params0)


# %load_ext tensorboard
# logdir = "logs/scratch/"
# writer = tf.summary.create_file_writer(logdir)
# tf.summary.trace_on(graph = True, profiler = True)

# mc.simulate_experiment(10)
res = mc.fit_simulated_experiments(num_samples = 100
    , num_experiments = 100
    , num_steps = tf.Variable(2000)
    , params = linear_params0
    , loss = msq_loss # could be sq_loss, cause it is automatically reduced
    # , optimizer = optimizer
    )


# use to overlay with theoretical cloud
# theoretical_tmp = (W0.numpy().T + 0.1*np.random.multivariate_normal(np.zeros(2), 9*np.linalg.inv(X_cov), size = 100))
m = ravel_dicts([linear_params0])
X_cov_ext = np.zeros((3,3)).astype(np.float32)
X_cov_ext[:2,:2] = X_cov # used in the actual simulation scheme
X_cov_ext[2, 2] = 1
cov_th = 9*0.01*np.linalg.inv(X_cov_ext) # 9 from noise squared, 0.01 from 100 samples in the simulation
tmp_x, tmp_y = mc.simulation_scheme(100)
cov_est = mc.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, linear_params0, loss = sq_loss)

proj = np.eye(3,2)#[[2,0,1]]


plot_confidence_2d_projection(m.reshape(-1,1), cov_est, proj, ravel_dicts(res))
plot_confidence_2d_projection(m.reshape(-1,1), cov_th, proj, ravel_dicts(res))



## Part II. Weighted Least Squares

W1 = tf.Variable([[-.5],[.8]], dtype = tf.float32)
b1 = tf.Variable(1.0, dtype = tf.float32)
linear_params1 = OrderedDict({'W': W1, 'b': b1})
# linear_params1_ = OrderedDict({'W': tf.Variable([[0.],[0.]], dtype = tf.float32), 'b': tf.Variable(0.0, dtype = tf.float32)})

mcw = ModelFromConcreteFunction(linear_equation, model_name = "test_w_model", params = linear_params1)
mcw.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
mcw.loss = sq_loss

X_cov = np.array([[1,.95], [.95,1]], dtype = np.float32)
mcw.simulation_scheme = linear_gaussian_simulation_scheme(
    np.array([0., 0.], dtype = np.float32),
    X_cov,
    linear_params1,
    y_ssq = 9
    )

weights = np.concatenate([np.repeat(1., 10).astype(np.float32), np.repeat(0., 90).astype(np.float32)])
weights = np.random.chisquare(2, 100).astype(np.float32)


res = mcw.fit_simulated_experiments(num_samples = 100
    , num_experiments = 100
    , num_steps = tf.Variable(2000)
    , params = linear_params1
    , weights = weights
    , loss = sq_loss #
    # , optimizer = optimizer
    )


tmp_x, tmp_y = mcw.simulation_scheme(100)
mcw.dldw2_plug_in_estimator(tmp_x, tmp_y, linear_params1, weights = weights, loss = sq_loss)
mcw.d2ld2w_plug_in_estimator(tmp_x, tmp_y, linear_params1, weights = weights, loss = sq_loss)
cov_tmp = mcw.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, linear_params1, weights = weights, loss = sq_loss)


m = ravel_dicts([linear_params1])
tmp_x, tmp_y = mcw.simulation_scheme(100)
cov_est = mcw.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, linear_params1, weights = weights, loss = sq_loss)

proj = np.eye(3,2)

plot_confidence_2d_projection(m.reshape(-1,1), cov_est, proj, ravel_dicts(res))




## Part III. Model with nonlinear functional equation

@tf.function
def scurve_equation(input_features, params):
    return params['b'] + params['a']*tf.sigmoid(tf.squeeze(tf.matmul(input_features, params['W'])) + params['s'])


a2 = tf.Variable(2., dtype = tf.float32)
b2 = tf.Variable(-.5, dtype = tf.float32)
s2 = tf.Variable(.0, dtype = tf.float32)
W2 = tf.Variable([[-.5],[.8]], dtype = tf.float32)

scurve_params2 = OrderedDict({'a': a2, 'b': b2, 's': s2, 'W': W2})

mcs = ModelFromConcreteFunction(scurve_equation, model_name = "test_s_model", params = scurve_params2)
mcs.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
mcs.loss = sq_loss


# interesting fact:
# it is instructive to see how misleading results could be if small covariance is picked
# then the model becomes dominated by noise
X_cov = np.array([[10,9.5], [9.5,10]], dtype = np.float32)
mcs.simulation_scheme = functional_gaussian_simulation_scheme(
    np.array([0., 0.], dtype = np.float32),
    X_cov,
    scurve_params2,
    y_ssq = 9,
    functional_equation = scurve_equation
    )


# mcs.fit(
#     tmp_x, tmp_y
#     , num_steps = tf.Variable(2000)
#     , loss = sq_loss
#     , params = scurve_params2)


# mc.simulate_experiment(10)
res = mcs.fit_simulated_experiments(num_samples = 1000
    , num_experiments = 100
    , num_steps = tf.Variable(2000)
    , params = scurve_params2
    , loss = sq_loss # could be sq_loss, cause it is automatically reduced
    # , optimizer = optimizer
    )


tmp_x, tmp_y = mcs.simulation_scheme(1000)

m = ravel_dicts([scurve_params2])
cov_est = mcs.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, scurve_params2, loss = sq_loss)

proj = np.eye(3,2)[[2,2,2,0,1]]

plot_confidence_2d_projection(m.reshape(-1,1), cov_est, proj, ravel_dicts(res))



## Part IV. Model with heteroskedasticity and Log-likelihood estimation

@tf.function
def ll_gaussian_loss(labels, predictions, params):
    # note the sign is flipped for minimization problem
    # note constant = .5*tf.math.log(2*np.pi) is removed from the result
    ssq = tf.square(params['sigma'])
    return .5*tf.math.log(ssq) + .5*tf.square(labels - predictions)/ssq


W3 = tf.Variable([[-.5],[.8]], dtype = tf.float32)
b3 = tf.Variable(1.0, dtype = tf.float32)
sigma = tf.Variable(3.0, dtype = tf.float32)
linear_params3 = OrderedDict({'W': W3, 'b': b3, 'sigma': sigma})

mcl = ModelFromConcreteFunction(linear_equation, model_name = "test_l_model", params = linear_params3)
mcl.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
mcl.loss = ll_gaussian_loss

X_cov = np.array([[10, 9.5], [9.5,10]], dtype = np.float32)
mcl.simulation_scheme = linear_gaussian_simulation_scheme(
    np.array([0., 0.], dtype = np.float32),
    X_cov,
    linear_params3,
    y_ssq = sigma.numpy()**2
    )

tmp_x, tmp_y = mcl.simulation_scheme(100)
mcl.dldw2_plug_in_estimator(tmp_x, tmp_y, linear_params3, loss = ll_gaussian_loss)
mcl.d2ld2w_plug_in_estimator(tmp_x, tmp_y, linear_params3, loss = ll_gaussian_loss)
cov_tmp = mcl.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, linear_params3, loss = ll_gaussian_loss)



# mc.simulate_experiment(10)
res = mcl.fit_simulated_experiments(num_samples = 1000
    , num_experiments = 100
    , num_steps = tf.Variable(2000)
    , params = linear_params3
    , loss = ll_gaussian_loss # could be sq_loss, cause it is automatically reduced
    # , optimizer = optimizer
    )


# use to overlay with theoretical cloud
# theoretical_tmp = (W0.numpy().T + 0.1*np.random.multivariate_normal(np.zeros(2), 9*np.linalg.inv(X_cov), size = 100))
m = ravel_dicts([linear_params3])
tmp_x, tmp_y = mcl.simulation_scheme(1000)
cov_est = mcl.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, linear_params3, loss = ll_gaussian_loss)

proj = np.eye(3,2)[[2,2,0,1]]


plot_confidence_2d_projection(m.reshape(-1,1), cov_est, proj, ravel_dicts(res))
