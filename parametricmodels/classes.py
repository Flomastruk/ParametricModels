import tqdm

import numpy as np
import tensorflow as tf
from collections import OrderedDict
# print(tf.__version__)

# from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.probability import MultivariateNormalFullCovariance

from parametricmodels.utils import tf_ravel_dict, params_ravel_to_dict, ravel_inputs, ravel_loss, tensor_to_numpy_dict, numpy_ravel, ravel_dicts

# TODO: define module-level Exceptions
# TODO: dummy method ParametricModel.predict returns params, this is illogical
# TODO: loss compatibility (instead of user difined loss functions)
# TODO: put example functions into a separate file
# TODO: integrate tensorflow-based distributions

class Error(Exception):
    '''Base class for exceptions in this module.'''
    pass

class NoPredictionMethodError(Error):
    '''Exception raised when no prediction method is supplied'''

    def __init__(self, message):
        self.message = message

class NoparamsError(Error):
    '''Exception raised when no prediction method is supplied'''

    def __init__(self, message):
        self.message = message


class ParametricModel():
    '''
    Dummy class with placeholder definitions
    '''

    def __init__(self, model_name = None, params = None):
        self.name = model_name
        if params is not None:
            self.params = params

    def fit(self, data):
        pass

    def predict(self, input_features, params = None):

        exception_message = 'No prediction method is specified for the model'
        if self.name is not None:
            exception_message += f' {self.name}'
        raise NoPredictionMethodError(exception_message)

    def get_params(self, params = None):
        if params is None:
            try:
                params = self.params
            except Exception as e:
                exception_message = 'No params were passed or pre-defined for predict method of the model'
                if self.name is not None:
                    exception_message += ' {}'.format(self.name)
                raise NoparamsError(exception_message)
        return params

    def simulate(self, n_simulations):
        pass

    def save(self, file_name = None):
        pass

    def create_param_dictionary(self, param_values = None):
        '''
        creates a dictionary of param variables that have same key-shape pairs as the model
        '''
        if param_values is None:
            param_values = {key: val.numpy() for key, val in self.params.items()}
        return ({key:tf.Variable(val if isinstance(val, np.ndarray) else val.numpy()) for key, val in param_values.items()})

    def assign_params(self, params, values):
        for key, val in params.items():
            if key in values:
                val.assign(values[key])


class ModelFromConcreteFunction(ParametricModel):

    def __init__(self, functional_equation, model_name = None, loss = None, loss_from_features = None, params = None, simulation_scheme = None):
        super().__init__(model_name = model_name, params = params)
        self.functional_equation = functional_equation

        if loss is not None:
            self.loss = loss
        if loss_from_features is not None:
            self.loss_from_features = loss_from_features
        if simulation_scheme is not None:
            self.simulation_scheme = simulation_scheme

    def predict(self, input_features, params = None):
        params = super().get_params(params = params)

        return self.functional_equation(input_features, params)


    def __call__(self, input_features, params = None):
        '''
        Equivalent to predict
        '''
        return self.predict(input_features, params)

    # interesting note: if I wrap it this way and then
    # recreate an optimizer, it would raise an exception:
    # ValueError: tf.function-decorated function tried to create variables on non-first call.
    @tf.function
    def train_step(self, input_features, input_labels,
        params, loss, loss_from_features, optimizer,
        apply_gradients = True, weights = None, tot_weight = None):

        # TODO:??? uniform treatment for all models
        # optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-2)
        print('Retracing train step')

        with tf.GradientTape() as tape:
            predictions = self.predict(input_features, params = params)
            if loss_from_features:
                losses = loss(input_labels, predictions, input_features, params)
            else:
                losses = loss(input_labels, predictions, params)

            if weights is None:
                objective = tf.reduce_mean(losses)
            else:
                assert input_labels.shape[0] == weights.shape[0], 'Weights must have same shape as input labels'
                objective = tf.reduce_mean(weights*losses)

        # TODO: create a mask with trainable params
        trainable_params = list(params.values())
        gradients = tape.gradient(objective, trainable_params)

        if apply_gradients:
            optimizer.apply_gradients(zip(gradients, trainable_params))

        if weights is None:
            return objective # train step returns current loss value
        else:  # if weights provided, renormalize post optimization step
            return objective*weights.shape[0]/(tf.reduce_sum(weights) if tot_weight is None else tot_weight)

    @tf.function
    def fit(self, input_features, input_labels, num_steps,
        params = None, loss = None, loss_from_features = None, weights = None,
        optimizer = None, verbose = True):
        '''
        Fit the model based on a set of input features and labels
        `loss_from_features` if True, then loss is passed `input_features` not predictions
        '''
        # TODO: incorporate stopping times

        if params is None:
            params = self.params
        if loss is None:
            loss = self.loss
        if loss_from_features is None and hasattr(self, 'loss_from_features'):
            loss_from_features = self.loss_from_features
        elif loss_from_features is None:
            loss_from_features = False
        if optimizer is None:
            optimizer = self.optimizer

        print('Retracing fit method')
        if weights is None:
            for t in range(num_steps):
                # for t in tqdm.tqdm(range(num_steps), disable = not verbose): ## does not support input tensors
                self.train_step(input_features, input_labels,
                    params = params, loss = loss, loss_from_features = loss_from_features, optimizer = optimizer)
        else:
            tot_weight = tf.reduce_sum(weights)
            for t in range(num_steps):
                self.train_step(input_features, input_labels,
                    params = params, loss = loss, loss_from_features = loss_from_features, optimizer = optimizer,
                    weights = weights, tot_weight = tot_weight)

        # if verbose:
        #     log_str = "Training complete, final loss: {0:.5f}".format(self.train_step(input_features, input_labels, params = params, loss = loss, optimizer = optimizer, apply_gradients = False).numpy())
        #     print(log_str)

        return params

    def simulate_experiment(self, num_samples):
        '''
        returns an array that comes from assigned simulation_scheme
        '''
        if self.simulation_scheme is None:
            return
        return self.simulation_scheme(num_samples)

    def fit_simulated_experiments(self, num_samples, num_experiments, num_steps,
        params = None, weights = None, loss = None, loss_from_features = None,
        optimizer = None, verbose = True):
        '''
        `num_samples` number of training samples per experiment
        `num_experiments` number of estimation experiments simulated
        `num_steps` number of steps optimizer is applied for each experiment
        `params` initial guess passed to optimization (optional if model has `params` attribute)
        '''
        res = []
        params = self.params if params is None else params

        params_ = self.create_param_dictionary(params)
        print("Begin fitting simulated experiments...")
        for ix_experiment in tqdm.tqdm(range(num_experiments), disable = not verbose):
            # reset params to the initial guess
            self.assign_params(params_, params)
            sample_input, sample_labels = self.simulate_experiment(num_samples)
            self.fit(sample_input, sample_labels, num_steps
                , params =  params_, loss = loss, loss_from_features = loss_from_features
                , weights = weights, verbose = False)
            res.append(tensor_to_numpy_dict(params_))

        return res

    def dldw2_plug_in_estimator(self, input_features, input_labels, params = None, weights = None
        , loss = None, loss_from_features = None):

        if params is None:
            params = self.params
        if loss is None:
            loss = self.loss
        if loss_from_features is None and hasattr(self, 'loss_from_features'):
            loss_from_features = self.loss_from_features
        elif loss_from_features is None:
            loss_from_features = False

        with tf.GradientTape() as tape:
            params_ = tf_ravel_dict(params)
            functional_equation_ = ravel_inputs(self.functional_equation, params)
            loss_ = ravel_loss(loss, params, loss_from_features = loss_from_features)

            predictions = functional_equation_(input_features, params_)
            if loss_from_features:
                # print(loss_from_features)
                # print(loss_)
                losses = loss_(input_labels, predictions, input_features, params_) # note --- loss always has reduced_mean across samples
            else:
                losses = loss_(input_labels, predictions, params_)

        jac = tape.jacobian(losses, params_)

        if weights is None:
            return tf.tensordot(jac, jac, axes = [[0], [0]])/input_labels.shape[0] # outer product of first derivatives
        else:
            assert jac.shape[0] == weights.shape[0]
            # TODO: think about theoretical framework. What is the proper way to normalize this and what is the meaning
            jac_w = tf.reshape(weights, (-1,1))*jac
            return tf.tensordot(jac_w, jac_w, axes = [[0], [0]])


    def d2ld2w_plug_in_estimator(self, input_features, input_labels, params = None, weights = None
        , loss = None, loss_from_features = None):

        if params is None:
            params = self.params
        if loss is None:
            loss = self.loss
        if loss_from_features is None and hasattr(self, 'loss_from_features'):
            loss_from_features = self.loss_from_features
        elif loss_from_features is None:
            loss_from_features = False

        # this might cause duplication when multiple estimators are caused and inputs are ravelled
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                params_ = tf_ravel_dict(params)
                functional_equation_ = ravel_inputs(self.functional_equation, params)
                loss_ = ravel_loss(loss, params, loss_from_features = loss_from_features)

                predictions = functional_equation_(input_features, params_)
                if weights is None:
                    if loss_from_features:
                        losses = loss_(input_labels, predictions, input_features, params_)
                    else:
                        losses = loss_(input_labels, predictions, params_)
                else:
                    if loss_from_features:
                        losses = weights*loss_(input_labels, predictions, input_features, params_)
                    else:
                        losses = weigths*loss_(input_labels, predictions, params_)

            dldw = tape2.gradient(losses, params_)/input_labels.shape[0] if weights is None else tape2.gradient(losses, params_)

        d2ldw2 = tape1.jacobian(dldw, params_)
        return d2ldw2


    @tf.function
    def parameter_covariance_plug_in_estimator(self, input_features, input_labels, params = None, weights = None
        , loss = None, loss_from_features = None):
        print("Retracing covariance estimation")

        if params is None:
            params = self.params
        if loss is None:
            loss = self.loss

        dldw2 = self.dldw2_plug_in_estimator(input_features, input_labels, params = params, weights = weights
            , loss = loss, loss_from_features = loss_from_features)
        d2ld2w_inv = tf.linalg.inv(self.d2ld2w_plug_in_estimator(input_features, input_labels, params = params, weights = weights
            , loss = loss, loss_from_features = loss_from_features))

        return (d2ld2w_inv @ dldw2 @ d2ld2w_inv)/input_labels.shape[0] if weights is None else (d2ld2w_inv @ dldw2 @ d2ld2w_inv)
