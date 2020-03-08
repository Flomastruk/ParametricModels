
import numpy as np

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
