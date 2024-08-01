import pytest
import numpy as np
import jax.numpy as jnp
from scqubits import backend_change
from scqubits.core import diag
from qutip import Qobj

def test_numpy_backend():
    backend_change.set_backend('numpy')
    assert isinstance(backend_change.backend, backend_change.NumpyBackend)
    
    evecs = np.array([[1, 0], [0, 1]], dtype=complex)
    matrix_qobj = Qobj(evecs)
    result = diag._convert_evecs_to_qobjs(evecs, matrix_qobj)
    assert len(result) == 2
    assert isinstance(result[0], Qobj)

def test_jax_backend():
    backend_change.set_backend('jax')
    assert isinstance(backend_change.backend, backend_change.JaxBackend)
    
    evecs = jnp.array([[1, 0], [0, 1]], dtype=complex)
    matrix_qobj = Qobj(np.array(evecs))
    result = diag._convert_evecs_to_qobjs(evecs, matrix_qobj)
    assert len(result) == 2
    assert isinstance(result[0], Qobj)
    # 确保结果是列表
    assert isinstance(result, list)

def test_evals_jax_dense_numpy_backend():
    backend_change.set_backend('numpy')
    matrix = np.array([[1, 2], [2, 1]], dtype=complex)
    evals = diag.evals_jax_dense(matrix, 1)
    assert len(evals) == 1

def test_evals_jax_dense_jax_backend():
    backend_change.set_backend('jax')
    matrix = jnp.array([[1, 2], [2, 1]], dtype=complex)
    evals = diag.evals_jax_dense(matrix, 1)
    assert len(evals) == 1

def test_esys_jax_dense_numpy_backend():
    backend_change.set_backend('numpy')
    matrix = np.array([[1, 2], [2, 1]], dtype=complex)
    evals, evecs = diag.esys_jax_dense(matrix, 1)
    assert len(evals) == 1
    assert evecs.shape[1] == 1
    if isinstance(evecs, np.ndarray):
        assert isinstance(evecs[0], np.ndarray)
    else:
        assert isinstance(evecs[0], Qobj)

def test_esys_jax_dense_jax_backend():
    backend_change.set_backend('jax')
    matrix = jnp.array([[1, 2], [2, 1]], dtype=complex)
    evals, evecs = diag.esys_jax_dense(matrix, 1)
    assert len(evals) == 1
    assert evecs.shape[1] == 1
    if isinstance(evecs, jnp.ndarray):
        assert isinstance(evecs[0], jnp.ndarray)
    else:
        assert isinstance(evecs[0], Qobj)

if __name__ == '__main__':
    pytest.main()
