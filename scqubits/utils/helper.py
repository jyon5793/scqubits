import itertools
import jax.numpy as jnp
from scqubits import backend_change as bc
import numpy as np

def ndindex(shape):
    return itertools.product(*[range(dim) for dim in shape])

@staticmethod
def convert_to_jax_compatible(array):
    if bc.backend.__name__ == "jax":
        return jnp.array([jnp.asarray(item) for item in array])
    return array

@staticmethod
def create_empty_array(length):
    if bc.backend.__name__ == "jax":
        return np.empty((length,), dtype=object) 
    else:
        return np.empty((length,), dtype=object)