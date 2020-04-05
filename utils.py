
import numpy as np
import tensorflow as tf


def tf_ravel_dict(tensor_dict):
    """
    output one dimensional tensor
    """
    return tf.concat([tf.reshape(w, [-1]) for w in tensor_dict.values()], axis = -1)


def weights_ravel_to_dict(weights_ravel, signature_weights):
    shapes = [w.shape for w in signature_weights.values()] # product of dimentions

    slice_indices = [np.prod(sh, dtype = np.int32) for sh in shapes]
    slice_indices.insert(0, 0)
    slice_indices = np.cumsum(slice_indices)

    weights_list = [tf.reshape(weights_ravel[i:j], sh) for i, j, sh in zip(slice_indices[:-1], slice_indices[1:], shapes)]
    return {key: w for key, w in zip(signature_weights.keys(), weights_list)}

# TODO: better: use the signature of functional equation
def ravel_inputs(functional_equation, signature_weights):

    def functional_equation_ravel(input_labels, weights_ravel):
        weights = weights_ravel_to_dict(weights_ravel, signature_weights)
        return functional_equation(input_labels, weights)

    return functional_equation_ravel


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
