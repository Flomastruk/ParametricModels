
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
def linear_equation(input_features, weights):
    return tf.squeeze(tf.matmul(input_features, weights['W']))+ weights['b']

@tf.function
def msq_loss(labels, predictions):
    return tf.reduce_mean(tf.square(labels - predictions))

@tf.function
def sq_loss(labels, predictions):
    return tf.square(labels - predictions)

#@tf.function
def linear_gaussian_simulation_scheme(X_mean, X_cov, weights, y_ssq):
    # @tf.function
    def concrete_linear_simulation_scheme(num_samples):
        X_sim = np.random.multivariate_normal(mean = X_mean, cov = X_cov, size = num_samples).astype(np.float32)
        err = np.random.normal(loc = 0., size = num_samples, scale = np.sqrt(y_ssq)).astype(np.float32)
        y_sim = linear_equation(X_sim, weights) + err

        return tf.Variable(X_sim) , y_sim
    return concrete_linear_simulation_scheme

# W = tf.Variable([[-1],[1]], dtype = tf.float32)
# b = tf.Variable(2, dtype = tf.float32)
# linear_weights = {'W': W, 'b': b}

# some sample-data
# tmp_n_samples = 100
# sample_input = np.matrix(np.arange(tmp_n_samples*2).reshape(tmp_n_samples,2).astype(np.float32))
# sample_input[:,1] = np.tile([-1.,1], tmp_n_samples//2 + 1)[:tmp_n_samples][:,np.newaxis]
# sample_labels = (sample_input[:,0] + sample_input[:,1] - 1).ravel() # 1, -1, 1


# mc.fit(sample_input, sample_labels, 10, weights = linear_weights, loss = msq_loss, optimizer = optimizer)

W0 = tf.Variable([[-.5],[.8]], dtype = tf.float32)
b0 = tf.Variable(1.0, dtype = tf.float32)
linear_weights0 = OrderedDict({'W': W0, 'b': b0})

mc = ModelFromConcreteFunction(linear_equation, model_name = "test_c_model", weights = linear_weights0)
mc.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-2)
mc.loss = msq_loss

X_cov = np.array([[1,.95], [.95,1]], dtype = np.float32)
mc.simulation_scheme = linear_gaussian_simulation_scheme(
    np.array([0., 0.], dtype = np.float32),
    X_cov,
    linear_weights0,
    y_ssq = 9
    )

# %load_ext tensorboard
# logdir = "logs/scratch/"
# writer = tf.summary.create_file_writer(logdir)
# tf.summary.trace_on(graph = True, profiler = True)

tmp_x, tmp_y = mc.simulation_scheme(100)
mc.fit(
    tmp_x, tmp_y
    , num_steps = tf.Variable(1200)
    , loss = sq_loss
    , weights = linear_weights0)

mc.dldw2_plug_in_estimator(tmp_x, tmp_y, linear_weights0, loss = sq_loss)
mc.d2ld2w_plug_in_estimator(tmp_x, tmp_y, linear_weights0, loss = sq_loss)
cov_tmp = mc.parameter_covariance_plug_in_estimator(tmp_x, tmp_y, linear_weights0, loss = sq_loss)




# mc.simulate_experiment(10)
res = mc.fit_simulated_experiments(num_samples = 100
    , num_experiments = 100
    , num_steps = tf.Variable(20000)
    , weights = linear_weights0
    , loss = msq_loss
    # , optimizer = optimizer
    )



# estimated confidence region
ux = np.linspace(*ax.get_xlim(), 100)
uy = np.linspace(*ax.get_ylim(), 100)
xs, ys = np.meshgrid(ux, uy)
xy = np.concatenate([xs.reshape(-1,1), ys.reshape(-1,1)], axis = -1)


def quadratic_form(m, V):
    def quad(z):
        z0 = z - m.reshape(1, -1)
        V_inv = np.linalg.inv(V)

        # print(z0.shape, V_inv.shape)
        return np.sum(z0 *np.matmul(z0, V_inv), axis = 1)
    return quad

quad = quadratic_form(linear_weights0['W'].numpy(), cov_tmp[:2, :2])
from scipy.stats import chi2


tmp = ravel_dicts(res)
theoretical_tmp = (W0.numpy().T + 0.1*np.random.multivariate_normal(np.zeros(2), 9*np.linalg.inv(X_cov), size = 100))


fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(tmp[:,0], tmp[:,1], alpha = 0.5)
ax.scatter(theoretical_tmp[:,0],theoretical_tmp[:,1], alpha = 0.5)
ax.contour(xs, ys, quad(xy).reshape(100,100), chi2(df = 2).ppf([0.95, 0.975, 0.995, 0.9995]), colors  = 'red')





# ### derivatives all around
#
# with tf.GradientTape() as tape:
#     linear_weights0_ = tf_ravel_dict(linear_weights0)
#     linear_equation_ = ravel_inputs(linear_equation, linear_weights0)
#
#     tmp_yh = linear_equation_(tmp_x, linear_weights0_)
#     L = sq_loss(tmp_yh, tmp_y)
#
#
# tmp_g = tape.jacobian(L, linear_weights0_)
# tmp_g
#
#
# dldw2 = tf.tensordot(tmp_g, tmp_g, axes = [[0], [0]])/tmp_x.shape[0]
# # tf.reduce_sum((tf.expand_dims(tmp_g, 1)*tf.expand_dims(tmp_g, -1)), axis = 0)/tmp_x.shape[0]
#
#
# with tf.GradientTape() as tape1:
#     with tf.GradientTape() as tape2:
#         linear_weights0_ = tf_ravel_dict(linear_weights0)
#         linear_equation_ = ravel_inputs(linear_equation, linear_weights0)
#
#         tmp_yh = linear_equation_(tmp_x, linear_weights0_)
#         L = sq_loss(tmp_yh, tmp_y)
#     dldw = tape2.gradient(L, linear_weights0_)/tmp_x.shape[0]
#
# d2ldw2 = tape1.jacobian(dldw, linear_weights0_)
#

# d2ldw2inv = tf.linalg.inv(d2ldw2)
#
# d2ldw2inv @ dldw2 @ d2ldw2inv
#
