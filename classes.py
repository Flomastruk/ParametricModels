
import tqdm

import numpy as np
import tensorflow as tf
# print(tf.__version__)

from tensorflow.keras.losses import MeanSquaredError
# from tensorflow.probability import MultivariateNormalFullCovariance

from utils import tensor_to_numpy_dict, numpy_ravel, ravel_dicts

# TODO: define module-level Exceptions
# TODO: dummy method ParametricModel.predict returns weights, this is illogical
# TODO: loss compatibility (instead of user difined loss functions)
# TODO: put example functions into a separate file
# TODO: integrate tensorflow-based distributions


class ParametricModel():
    """
    Dummy class with placeholder definitions
    """

    def __init__(self, model_name = None, weights = None):
        self.name = model_name
        if weights is not None:
            self.weights = weights

    def fit(self, data):
        pass

    def predict(self, input_features, weights = None):

        if weights is None:
            try:
                weights = self.weights
            except Exception as e:
                exception_message = 'No weights were passed or pre-defined for predict method of the model'
                if self.name is not None:
                    exception_message += ' ' + "'{}'".format(self.name)
                print(exception_message)
                raise e
        return weights

    def simulate(self, n_simulations):
        pass

    def save(self, file_name = None):
        pass

    def create_weight_dictionary(self, weight_values = None):
        """
        creates a dictionary of weight variables that have same key-shape pairs as the model
        """
        if weight_values is None:
            weight_values = {key: val.numpy() for key, val in self.weights.items()}
        return ({key:tf.Variable(val if isinstance(val, np.ndarray) else val.numpy()) for key, val in weight_values.items()})

    def assign_weights(self, weights, values):
        for key, val in weights.items():
            if key in values:
                val.assign(values[key])


class ModelFromConcreteFunction(ParametricModel):

    def __init__(self, functional_equation, model_name = None, loss = None, weights = None, simulation_scheme = None):
        self.functional_equation = functional_equation
        self.simulation_scheme = simulation_scheme

        super().__init__(model_name = model_name, weights = weights)

        if loss is not None:
            self.loss = loss

    def predict(self, input_features, weights = None):
        weights = super().predict(input_features, weights)

        return self.functional_equation(input_features, weights)

    def __call__(self, input_features, weights = None):
        """
        Equivalent to predict
        """
        return self.predict(input_features, weights)

    # interesting note: if I wrap it this way and then
    # recreate an optimizer, it would raise exception:
    # ValueError: tf.function-decorated function tried to create variables on non-first call.
    @tf.function
    def train_step(self, input_features, input_labels, weights = None, loss = None, optimizer = None, apply_gradients = True):

        # TODO:??? uniform treatment for all models
        # optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-2)
        if weights is None:
            weights = self.weights

        with tf.GradientTape() as tape:
            predictions = self.predict(input_features, weights = weights)
            # print(predictions)
            objective = loss(input_labels, predictions)

        trainable_weights = list(weights.values())
        gradients = tape.gradient(objective, trainable_weights)
        optimizer.apply_gradients(zip(gradients, trainable_weights))

        return objective # train step returns current loss value


    def fit(self, input_features, input_labels, num_steps, weights = None, loss = None, optimizer = None, verbose = True):
        """
        Fit the model based on a set of input features and labels
        """
        # TODO: incorporate stopping times
        # TODO: pipelines? should be a part of input_features/input_lables -- whatever is accepted by an API/function input
        if weights is None:
            weights = self.weights
        if loss is None:
            loss = self.loss
        if optimizer is None:
            optimizer = self.optimizer

        for t in tqdm.tqdm(range(num_steps), disable = not verbose):
            self.train_step(input_features, input_labels, weights = weights, loss = loss, optimizer = optimizer)
            # print(weights)

        # print("Current Predictions:\n")
        # print(self(input_features))
        if verbose:
            log_str = "Training complete, final loss: {0:.5f}".format(self.train_step(input_features, input_labels, weights = weights, loss = loss, optimizer = optimizer, apply_gradients = False).numpy())
            print(log_str)

        return weights

    def simulate_experiment(self, num_samples):
        """
        returns an array that comes from assigned simulation_scheme
        """
        if self.simulation_scheme is None:
            return
        return self.simulation_scheme(num_samples)

    def fit_simulated_experiments(self, num_samples, num_experiments, num_steps, weights = None, loss = None, optimizer = None, verbose = True):
        res = []
        weights = self.weights if weights is None else weights

        weights_ = self.create_weight_dictionary(weights)
        print("Begin fitting simulated experiments...")
        for ix_experiment in tqdm.tqdm(range(num_experiments), disable = not verbose):
            # reset weights to the initial guess
            self.assign_weights(weights_, weights)
            sample_input, sample_labels = self.simulate_experiment(num_samples)
            self.fit(sample_input, sample_labels, num_steps, weights =  weights_, loss = loss, optimizer = optimizer, verbose = False)
            res.append(tensor_to_numpy_dict(weights_))

        return res

    def d1d1t_plug_in_estimator(self, input_features, input_lables, weights = None):
        with tf.GradientTape() as tape:
            predictions = self.predict(input_features)
            objective = loss(input_labels, predictions)
        pass

    def d2_plug_in(self, input_features, input_lables):
        pass
