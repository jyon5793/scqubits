import pytest
import numpy as np
import jax.numpy as jnp
import scipy
from scqubits import backend_change
from scqubits.core.discretization import Grid1d, band_matrix


def test_backend_switch_to_numpy():
    backend_change.set_backend('numpy')

    # 测试 numpy 后端
    array = np.array([1, 2, 3])
    result = array + 1
    assert isinstance(result, np.ndarray)
    assert np.all(result == np.array([2, 3, 4]))

    # 测试 Grid1d 类
    grid = Grid1d(min_val=0.0, max_val=1.0, pt_count=5)
    linspace = grid.make_linspace()
    assert isinstance(linspace, np.ndarray)
    assert np.allclose(linspace, np.linspace(0.0, 1.0, 5))

    first_derivative = grid.first_derivative_matrix()
    assert first_derivative.shape == (5, 5)

    second_derivative = grid.second_derivative_matrix()
    assert second_derivative.shape == (5, 5)


def test_backend_switch_to_jax():
    backend_change.set_backend('jax')

    # 测试 jax 后端
    array = jnp.array([1, 2, 3])
    result = array + 1
    assert isinstance(result, jnp.ndarray)
    assert np.all(result == jnp.array([2, 3, 4]))

    # 测试 Grid1d 类
    grid = Grid1d(min_val=0.0, max_val=1.0, pt_count=5)
    linspace = grid.make_linspace()
    assert isinstance(linspace, jnp.ndarray)
    assert np.allclose(linspace, jnp.linspace(0.0, 1.0, 5))

    first_derivative = grid.first_derivative_matrix()
    assert first_derivative.shape == (5, 5)

    second_derivative = grid.second_derivative_matrix()
    assert second_derivative.shape == (5, 5)


def test_backend_switch_between_numpy_and_jax():
    # 切换到 jax 后端
    backend_change.set_backend('jax')
    grid_jax = Grid1d(min_val=0.0, max_val=1.0, pt_count=5)
    linspace_jax = grid_jax.make_linspace()
    assert isinstance(linspace_jax, jnp.ndarray)
    assert np.allclose(linspace_jax, jnp.linspace(0.0, 1.0, 5))

    # 切换到 numpy 后端
    backend_change.set_backend('numpy')
    grid_numpy = Grid1d(min_val=0.0, max_val=1.0, pt_count=5)
    linspace_numpy = grid_numpy.make_linspace()
    assert isinstance(linspace_numpy, np.ndarray)
    assert np.allclose(linspace_numpy, np.linspace(0.0, 1.0, 5))


def test_band_matrix():
    backend_change.set_backend('numpy')
    coeffs = [1, 2, 3]
    offsets = [0, 1, -1]
    dim = 5
    mat_numpy = band_matrix(coeffs, offsets, dim)
    assert isinstance(mat_numpy, scipy.sparse.spmatrix)  # or scipy.sparse.spmatrix
    assert mat_numpy.shape == (dim, dim)

    backend_change.set_backend('jax')
    mat_jax = band_matrix(coeffs, offsets, dim)
    assert isinstance(mat_jax, scipy.sparse.spmatrix)  # or scipy.sparse.spmatrix
    assert mat_jax.shape == (dim, dim)


if __name__ == '__main__':
    pytest.main()