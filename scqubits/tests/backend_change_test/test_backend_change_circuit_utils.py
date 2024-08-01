import pytest
import numpy as np
import jax.numpy as jnp
from scqubits import backend_change

def test_backend_numpy():
    backend_change.set_backend('numpy')
    array = np.array([1, 2, 3])
    result = array + 1
    assert isinstance(result, np.ndarray)
    assert np.all(result == np.array([2, 3, 4]))

def test_backend_jax():
    backend_change.set_backend('jax')
    array = jnp.array([1, 2, 3])
    result = array + 1
    assert isinstance(result, jnp.ndarray)
    assert np.all(result == jnp.array([2, 3, 4]))

def test_backend_switch_to_numpy():
    backend_change.set_backend('jax')
    jax_array = jnp.array([1, 2, 3])
    jax_result = jax_array + 1
    assert isinstance(jax_result, jnp.ndarray)
    assert np.all(jax_result == jnp.array([2, 3, 4]))
    
    backend_change.set_backend('numpy')
    numpy_array = np.array([1, 2, 3])
    numpy_result = numpy_array + 1
    assert isinstance(numpy_result, np.ndarray)
    assert np.all(numpy_result == np.array([2, 3, 4]))

def test_backend_switch_to_jax():
    backend_change.set_backend('numpy')
    numpy_array = np.array([1, 2, 3])
    numpy_result = numpy_array + 1
    assert isinstance(numpy_result, np.ndarray)
    assert np.all(numpy_result == np.array([2, 3, 4]))
    
    backend_change.set_backend('jax')
    jax_array = jnp.array([1, 2, 3])
    jax_result = jax_array + 1
    assert isinstance(jax_result, jnp.ndarray)
    assert np.all(jax_result == jnp.array([2, 3, 4]))

if __name__ == '__main__':
    pytest.main()