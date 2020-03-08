

import os
os.chdir('../../Other/ParametricModels')
os.getcwd()

from classes import ModelFromConcreteFunction
import tensorflow as tf



@tf.function
def linear_equation(input_features, weights):
    return tf.squeeze(tf.matmul(input_features, weights['W']))+ weights['b']

@tf.function
def sq_loss(labels, predictions):
    return tf.reduce_mean(tf.square(labels - predictions))

#@tf.function
def linear_gaussian_simulation_scheme(X_mean, X_cov, weights, y_ssq):
    # @tf.function
    def concrete_linear_simulation_scheme(num_samples):
        X_sim = np.random.multivariate_normal(mean = X_mean, cov = X_cov, size = num_samples).astype(np.float32)
        err = np.random.normal(loc = 0., size = num_samples, scale = np.sqrt(y_ssq)).astype(np.float32)
        y_sim = linear_equation(X_sim, weights) + err

        return X_sim, y_sim

    return concrete_linear_simulation_scheme

W = tf.Variable([[-1],[1]], dtype = tf.float32)
b = tf.Variable(2, dtype = tf.float32)
linear_weights = {'W': W, 'b': b}

mc = ModelFromConcreteFunction(linear_equation, model_name = "test_c_model", weights = linear_weights)

# some sample-data
tmp_n_samples = 100
sample_input = np.matrix(np.arange(tmp_n_samples*2).reshape(tmp_n_samples,2).astype(np.float32))
sample_input[:,1] = np.tile([-1.,1], tmp_n_samples//2 + 1)[:tmp_n_samples][:,np.newaxis]
sample_labels = (sample_input[:,0] + sample_input[:,1] - 1).ravel() # 1, -1, 1

optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-2)
# mc.fit(sample_input, sample_labels, 10, weights = linear_weights, loss = sq_loss, optimizer = optimizer)


W0 = tf.Variable([[-.5],[.8]], dtype = tf.float32)
b0 = tf.Variable(1.0, dtype = tf.float32)
linear_weights0 = {'W': W0, 'b': b0}

mc.simulation_scheme = linear_gaussian_simulation_scheme(
    np.array([0., 0.], dtype = np.float32),
    np.eye(2, dtype = np.float32),
    linear_weights0,
    y_ssq = 9
    )


# mc.simulate_experiment(10)
res = mc.fit_simulated_experiments(num_samples = 100
    , num_experiments = 4
    , num_steps = 1000
    , weights = linear_weights0
    , loss = sq_loss
    , optimizer = optimizer)



ravel_dicts(res)
