
import os
import numpy as np
import tensorflow as tf

from collections import OrderedDict

from classes import ModelFromConcreteFunction
from utils import ravel_inputs, tf_ravel_dict, ravel_dicts


@tf.function
def linear_equation(input_features, params):
    return tf.squeeze(tf.matmul(input_features, params['W']))+ params['b']

# @tf.function
# def msq_loss(labels, predictions, params = None):
#     return tf.reduce_mean(tf.square(labels - predictions))

@tf.function
def sq_loss(labels, predictions, params = None):
    return tf.square(labels - predictions)


def functional_gaussian_simulation_scheme(X_mean, X_cov, params, y_ssq, functional_equation):
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
    , loss = sq_loss
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




## Part III. Model with non-linear functional equation

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
res = mcs.fit_simulated_experiments(num_samples = 10000
    , num_experiments = 100
    , num_steps = tf.Variable(2000)
    , params = scurve_params2
    , loss = sq_loss
    )


tmp_x, tmp_y = mcs.simulation_scheme(10000)

m = ravel_dicts([scurve_params2])
cov_est = mcs.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, scurve_params2, loss = sq_loss)

proj = np.eye(3,2)[[2,2,2,0,1]]

plot_confidence_2d_projection(m.reshape(-1,1), cov_est, proj, ravel_dicts(res))



## Part IV. Log-likelihood estimation

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


# mc.simulate_experiment(10)
res = mcl.fit_simulated_experiments(num_samples = 1000
    , num_experiments = 100
    , num_steps = tf.Variable(2000)
    , params = linear_params3
    , loss = ll_gaussian_loss # could be sq_loss, cause it is automatically reduced
    # , optimizer = optimizer
    )



m = ravel_dicts([linear_params3])
tmp_x, tmp_y = mcl.simulation_scheme(1000)
cov_est = mcl.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, linear_params3, loss = ll_gaussian_loss)

proj = np.eye(3,2)[[2,2,0,1]]


plot_confidence_2d_projection(m.reshape(-1,1), cov_est, proj, ravel_dicts(res))


# tmp_x, tmp_y = mcl.simulation_scheme(100)
# mcl.dldw2_plug_in_estimator(tmp_x, tmp_y, linear_params3, loss = ll_gaussian_loss)
# mcl.d2ld2w_plug_in_estimator(tmp_x, tmp_y, linear_params3, loss = ll_gaussian_loss)
# cov_tmp = mcl.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, linear_params3, loss = ll_gaussian_loss)

## Part V. Heteroskedastic model

@tf.function
def ll_heteroskedastic_gaussian_loss(labels, predictions, features, params):
    # note the sign is flipped for minimization problem
    # note constant = .5*tf.math.log(2*np.pi) is removed from the result
    ssq = tf.square(tf.squeeze(tf.matmul(features, params['T'])) + params['sigma'])
    return .5*tf.math.log(ssq) + .5*tf.square(labels - predictions)/ssq


def linear_heteroskedastic_gaussian_simulation_scheme(X_mean, X_cov, params):
    # @tf.function
    def concrete_linear_simulation_scheme(num_samples):
        X_sim = np.random.multivariate_normal(mean = X_mean, cov = X_cov, size = num_samples).astype(np.float32)
        err = np.random.normal(loc = 0., size = num_samples, scale = 1.).astype(np.float32)
        err*= tf.squeeze(X_sim @ params['T']) + params['sigma']
        y_sim = linear_equation(X_sim, params) + err

        return tf.Variable(X_sim) , y_sim
    return concrete_linear_simulation_scheme



W4 = tf.Variable([[-.5],[.8]], dtype = tf.float32)
b4 = tf.Variable(1.0, dtype = tf.float32)
T4 = tf.Variable([[1],[0]], dtype = tf.float32)
sigma4 = tf.Variable(3.0, dtype = tf.float32)
linear_params4 = OrderedDict({'W': W4, 'b': b4, 'T': T4, 'sigma': sigma4})


mch = ModelFromConcreteFunction(linear_equation, model_name = "test_h_model", params = linear_params4, loss_from_features = True)
mch.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
mch.loss = ll_heteroskedastic_gaussian_loss




X_cov = np.array([[10, 9.5], [9.5,10]], dtype = np.float32)
mch.simulation_scheme = linear_heteroskedastic_gaussian_simulation_scheme(
    np.array([10., 10.], dtype = np.float32),
    X_cov,
    linear_params4,
    )


# tmp_x, tmp_y = mch.simulation_scheme(100)
# mch.dldw2_plug_in_estimator(tmp_x, tmp_y, linear_params4, loss = ll_heteroskedastic_gaussian_loss)
# mch.d2ld2w_plug_in_estimator(tmp_x, tmp_y, linear_params4, loss = ll_heteroskedastic_gaussian_loss)
# cov_tmp = mch.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, linear_params4, loss = ll_heteroskedastic_gaussian_loss)


res = mch.fit_simulated_experiments(num_samples = 10000
    , num_experiments = 100
    , num_steps = tf.Variable(4000)
    , params = linear_params4
    , loss = ll_heteroskedastic_gaussian_loss
    )




m = ravel_dicts([linear_params4])
tmp_x, tmp_y = mch.simulation_scheme(10000)
mch.fit(tmp_x, tmp_y, num_steps = tf.Variable(4000), params = linear_params4, loss = ll_heteroskedastic_gaussian_loss)
cov_est = mch.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, linear_params4, loss = ll_heteroskedastic_gaussian_loss)
cov_est = mch.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, mch.params, loss = ll_heteroskedastic_gaussian_loss)

proj = np.eye(3,2)[[2,2,2,0,1,2]]
proj = np.eye(3,2)[[2,2,2,0,2,1]]
proj = np.eye(3,2)[[0,1,2,2,2,2]]
proj = np.eye(3,2)[[0,2,1,2,2,2]]


plot_confidence_2d_projection(m.reshape(-1,1), cov_est, proj, ravel_dicts(res))
plot_confidence_2d_projection(m.reshape(-1,1), cov_est, proj, ravel_dicts(res))
