import pytest
import numpy as np
from scqubits import backend_change
from scqubits.core.oscillator import Oscillator, KerrOscillator, harm_osc_wavefunction, convert_to_E_osc, convert_to_l_osc

@pytest.fixture
def osc_params():
    return {
        "E_osc": 5.0,
        "l_osc": 1.0,
        "truncated_dim": 10
    }

@pytest.fixture
def kerr_osc_params():
    return {
        "E_osc": 5.0,
        "K": 0.05,
        "l_osc": 1.0,
        "truncated_dim": 10
    }

def test_harm_osc_wavefunction():
    x = np.linspace(-5, 5, 100)
    backend_change.set_backend('numpy')
    numpy_result = harm_osc_wavefunction(0, x, 1.0)
    backend_change.set_backend('jax')
    jax_result = harm_osc_wavefunction(0, x, 1.0)
    assert np.allclose(numpy_result, jax_result, atol=1e-6)

def test_convert_to_E_osc():
    backend_change.set_backend('numpy')
    numpy_result = convert_to_E_osc(5.0, 2.0)
    backend_change.set_backend('jax')
    jax_result = convert_to_E_osc(5.0, 2.0)
    assert np.isclose(numpy_result, jax_result, atol=1e-6)

def test_convert_to_l_osc():
    backend_change.set_backend('numpy')
    numpy_result = convert_to_l_osc(5.0, 2.0)
    backend_change.set_backend('jax')
    jax_result = convert_to_l_osc(5.0, 2.0)
    assert np.isclose(numpy_result, jax_result, atol=1e-6)

def test_oscillator_eigenvals(osc_params):
    backend_change.set_backend('numpy')
    osc_numpy = Oscillator(**osc_params)
    numpy_result = osc_numpy.eigenvals()
    backend_change.set_backend('jax')
    osc_jax = Oscillator(**osc_params)
    jax_result = osc_jax.eigenvals()
    assert np.allclose(numpy_result, jax_result, atol=1e-6)

def test_oscillator_eigensys(osc_params):
    backend_change.set_backend('numpy')
    osc_numpy = Oscillator(**osc_params)
    numpy_evals, numpy_evecs = osc_numpy.eigensys()
    backend_change.set_backend('jax')
    osc_jax = Oscillator(**osc_params)
    jax_evals, jax_evecs = osc_jax.eigensys()
    assert np.allclose(numpy_evals, jax_evals, atol=1e-6)
    assert np.allclose(numpy_evecs, jax_evecs, atol=1e-6)

def test_oscillator_creation_operator(osc_params):
    backend_change.set_backend('numpy')
    osc_numpy = Oscillator(**osc_params)
    numpy_result = osc_numpy.creation_operator()
    backend_change.set_backend('jax')
    osc_jax = Oscillator(**osc_params)
    jax_result = osc_jax.creation_operator()
    assert np.allclose(numpy_result, jax_result, atol=1e-6)

def test_oscillator_annihilation_operator(osc_params):
    backend_change.set_backend('numpy')
    osc_numpy = Oscillator(**osc_params)
    numpy_result = osc_numpy.annihilation_operator()
    backend_change.set_backend('jax')
    osc_jax = Oscillator(**osc_params)
    jax_result = osc_jax.annihilation_operator()
    assert np.allclose(numpy_result, jax_result, atol=1e-6)

def test_oscillator_phi_operator(osc_params):
    backend_change.set_backend('numpy')
    osc_numpy = Oscillator(**osc_params)
    numpy_result = osc_numpy.phi_operator()
    backend_change.set_backend('jax')
    osc_jax = Oscillator(**osc_params)
    jax_result = osc_jax.phi_operator()
    assert np.allclose(numpy_result, jax_result, atol=1e-6)

def test_oscillator_n_operator(osc_params):
    backend_change.set_backend('numpy')
    osc_numpy = Oscillator(**osc_params)
    numpy_result = osc_numpy.n_operator()
    backend_change.set_backend('jax')
    osc_jax = Oscillator(**osc_params)
    jax_result = osc_jax.n_operator()
    assert np.allclose(numpy_result, jax_result, atol=1e-6)

def test_kerr_oscillator_eigenvals(kerr_osc_params):
    backend_change.set_backend('numpy')
    kerr_osc_numpy = KerrOscillator(**kerr_osc_params)
    numpy_result = kerr_osc_numpy.eigenvals()
    backend_change.set_backend('jax')
    kerr_osc_jax = KerrOscillator(**kerr_osc_params)
    jax_result = kerr_osc_jax.eigenvals()
    assert np.allclose(numpy_result, jax_result, atol=1e-6)

def test_kerr_oscillator_eigensys(kerr_osc_params):
    backend_change.set_backend('numpy')
    kerr_osc_numpy = KerrOscillator(**kerr_osc_params)
    numpy_evals, numpy_evecs = kerr_osc_numpy.eigensys()
    backend_change.set_backend('jax')
    kerr_osc_jax = KerrOscillator(**kerr_osc_params)
    jax_evals, jax_evecs = kerr_osc_jax.eigensys()
    assert np.allclose(numpy_evals, jax_evals, atol=1e-6)
    assert np.allclose(numpy_evecs, jax_evecs, atol=1e-6)

if __name__ == "__main__":
    pytest.main()