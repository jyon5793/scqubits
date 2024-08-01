import pytest
import numpy as np
import jax
import jax.numpy as jnp
from scqubits import backend_change
from scqubits.core.fluxonium import Fluxonium

@pytest.fixture
def fluxonium_params():
    return {
        "EJ": 8.9,
        "EC": 2.5,
        "EL": 0.5,
        "flux": 0.0,
        "cutoff": 110,
        "truncated_dim": 10,
    }

def test_backend_switch_to_numpy(fluxonium_params):
    backend_change.set_backend('numpy')

    fluxonium = Fluxonium(**fluxonium_params)

    assert isinstance(fluxonium.EJ, float)
    assert isinstance(fluxonium.EC, float)
    assert isinstance(fluxonium.EL, float)

    hamiltonian = fluxonium.hamiltonian()
    assert isinstance(hamiltonian, np.ndarray)
    assert hamiltonian.shape == (fluxonium_params["cutoff"], fluxonium_params["cutoff"])

    wavefunc = fluxonium.wavefunction(esys=None, which=0)
    assert isinstance(wavefunc.amplitudes, np.ndarray)
    assert wavefunc.amplitudes.shape[0] == fluxonium._default_grid.pt_count


def test_backend_switch_to_jax(fluxonium_params):
    backend_change.set_backend('jax')

    fluxonium = Fluxonium(**fluxonium_params)

    assert isinstance(fluxonium.EJ, float)
    assert isinstance(fluxonium.EC, float)
    assert isinstance(fluxonium.EL, float)

    hamiltonian = fluxonium.hamiltonian()
    assert isinstance(hamiltonian, jnp.ndarray)
    assert hamiltonian.shape == (fluxonium_params["cutoff"], fluxonium_params["cutoff"])

    wavefunc = fluxonium.wavefunction(esys=None, which=0)
    assert isinstance(wavefunc.amplitudes, jnp.ndarray)
    assert wavefunc.amplitudes.shape[0] == fluxonium._default_grid.pt_count


def test_backend_switch_between_numpy_and_jax(fluxonium_params):
    jax.config.update("jax_enable_x64", True)
    # 切换到 jax 后端
    backend_change.set_backend('jax')
    fluxonium_jax = Fluxonium(**fluxonium_params)

    hamiltonian_jax = fluxonium_jax.hamiltonian()
    wavefunc_jax = fluxonium_jax.wavefunction(esys=None, which=0)

    # 切换到 numpy 后端
    backend_change.set_backend('numpy')
    fluxonium_numpy = Fluxonium(**fluxonium_params)

    hamiltonian_numpy = fluxonium_numpy.hamiltonian()
    wavefunc_numpy = fluxonium_numpy.wavefunction(esys=None, which=0)

    assert np.allclose(hamiltonian_jax, hamiltonian_numpy, atol=1e-6)
    assert np.allclose(wavefunc_jax.amplitudes, wavefunc_numpy.amplitudes, atol=1e-6)


if __name__ == '__main__':
    pytest.main()
