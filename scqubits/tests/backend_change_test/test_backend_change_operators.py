import pytest
import numpy as np
import scipy.sparse as sp_sparse
import jax
import os
os.environ['JAX_ENABLE_X64'] = 'True'
jax.config.update("jax_enable_x64", True)
from scqubits import backend_change
from scqubits.core.operators import (
    annihilation, annihilation_sparse, creation, creation_sparse,
    hubbard_sparse, number, number_sparse, a_plus_adag_sparse, a_plus_adag,
    cos_theta_harmonic, sin_theta_harmonic, iadag_minus_ia_sparse,
    iadag_minus_ia, sigma_minus, sigma_plus, sigma_x, sigma_y, sigma_z
)

@pytest.fixture
def dimension():
    return 5

def test_annihilation(dimension):
    backend_change.set_backend('numpy')
    numpy_result = annihilation(dimension)
    backend_change.set_backend('jax')
    jax_result = annihilation(dimension)
    assert np.allclose(numpy_result, jax_result)

def test_annihilation_sparse(dimension):
    backend_change.set_backend('numpy')
    numpy_result = annihilation_sparse(dimension)
    backend_change.set_backend('jax')
    jax_result = annihilation_sparse(dimension)
    assert np.allclose(numpy_result.toarray(), jax_result.toarray())

def test_creation(dimension):
    backend_change.set_backend('numpy')
    numpy_result = creation(dimension)
    backend_change.set_backend('jax')
    jax_result = creation(dimension)
    assert np.allclose(numpy_result, jax_result)

def test_creation_sparse(dimension):
    backend_change.set_backend('numpy')
    numpy_result = creation_sparse(dimension)
    backend_change.set_backend('jax')
    jax_result = creation_sparse(dimension)
    assert np.allclose(numpy_result.toarray(), jax_result.toarray())

def test_hubbard_sparse(dimension):
    j1, j2 = 1, 2
    backend_change.set_backend('numpy')
    numpy_result = hubbard_sparse(j1, j2, dimension)
    backend_change.set_backend('jax')
    jax_result = hubbard_sparse(j1, j2, dimension)
    assert np.allclose(numpy_result.toarray(), jax_result.toarray())

def test_number(dimension):
    backend_change.set_backend('numpy')
    numpy_result = number(dimension)
    backend_change.set_backend('jax')
    jax_result = number(dimension)
    assert np.allclose(numpy_result, jax_result)

def test_number_sparse(dimension):
    backend_change.set_backend('numpy')
    numpy_result = number_sparse(dimension)
    backend_change.set_backend('jax')
    jax_result = number_sparse(dimension)
    assert np.allclose(numpy_result.toarray(), jax_result.toarray())

def test_a_plus_adag_sparse(dimension):
    backend_change.set_backend('numpy')
    numpy_result = a_plus_adag_sparse(dimension)
    backend_change.set_backend('jax')
    jax_result = a_plus_adag_sparse(dimension)
    assert np.allclose(numpy_result.toarray(), jax_result.toarray())

def test_a_plus_adag(dimension):
    backend_change.set_backend('numpy')
    numpy_result = a_plus_adag(dimension)
    backend_change.set_backend('jax')
    jax_result = a_plus_adag(dimension)
    assert np.allclose(numpy_result, jax_result)

def test_cos_theta_harmonic(dimension):
    backend_change.set_backend('numpy')
    numpy_result = cos_theta_harmonic(dimension)
    backend_change.set_backend('jax')
    jax_result = cos_theta_harmonic(dimension)
    assert np.allclose(numpy_result, jax_result)

def test_sin_theta_harmonic(dimension):
    backend_change.set_backend('numpy')
    numpy_result = sin_theta_harmonic(dimension)
    backend_change.set_backend('jax')
    jax_result = sin_theta_harmonic(dimension)
    assert np.allclose(numpy_result, jax_result)

def test_iadag_minus_ia_sparse(dimension):
    backend_change.set_backend('numpy')
    numpy_result = iadag_minus_ia_sparse(dimension)
    backend_change.set_backend('jax')
    jax_result = iadag_minus_ia_sparse(dimension)
    assert np.allclose(numpy_result.toarray(), jax_result.toarray())

def test_iadag_minus_ia(dimension):
    backend_change.set_backend('numpy')
    numpy_result = iadag_minus_ia(dimension)
    backend_change.set_backend('jax')
    jax_result = iadag_minus_ia(dimension)
    assert np.allclose(numpy_result, jax_result)

def test_sigma_minus():
    backend_change.set_backend('numpy')
    numpy_result = sigma_minus()
    backend_change.set_backend('jax')
    jax_result = sigma_minus()
    assert np.allclose(numpy_result, jax_result)

def test_sigma_plus():
    backend_change.set_backend('numpy')
    numpy_result = sigma_plus()
    backend_change.set_backend('jax')
    jax_result = sigma_plus()
    assert np.allclose(numpy_result, jax_result)

def test_sigma_x():
    backend_change.set_backend('numpy')
    numpy_result = sigma_x()
    backend_change.set_backend('jax')
    jax_result = sigma_x()
    assert np.allclose(numpy_result, jax_result)

def test_sigma_y():
    backend_change.set_backend('numpy')
    numpy_result = sigma_y()
    backend_change.set_backend('jax')
    jax_result = sigma_y()
    assert np.allclose(numpy_result, jax_result)

def test_sigma_z():
    backend_change.set_backend('numpy')
    numpy_result = sigma_z()
    backend_change.set_backend('jax')
    jax_result = sigma_z()
    assert np.allclose(numpy_result, jax_result)

if __name__ == "__main__":
    pytest.main()
