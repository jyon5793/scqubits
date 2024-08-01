import pytest
import numpy as np
from scqubits.backend_change import set_backend
from scqubits.core.noise import NoisySystem

# Mock NoisySystem class for testing
class MockNoisySystem(NoisySystem):
    def __init__(self):
        self.flux = 0.0
        self.EJ = 1.0
        self.EL = 0.5

    def eigensys(self, evals_count):
        evals = np.linspace(0, 10, evals_count)
        evecs = np.eye(evals_count)
        return evals, evecs

    def supported_noise_channels(self):
        return ["t1_flux_bias_line", "tphi_1_over_f_flux"]

    def set_and_return(self, attr_name, value):
        setattr(self, attr_name, value)
        return self

    def d_hamiltonian_d_flux(self):
        return np.eye(10)

    def t1_effective(self, **kwargs):
        return 1.0

    def t2_effective(self, **kwargs):
        return 2.0

@pytest.fixture
def noisy_system():
    return MockNoisySystem()

def test_noisy_system_numpy(noisy_system):
    set_backend('numpy')
    param_vals = np.linspace(-0.5, 0.5, 100)

    t1_vals = [noisy_system.t1_effective() for _ in param_vals]
    t2_vals = [noisy_system.t2_effective() for _ in param_vals]

    assert np.allclose(t1_vals, 1.0)
    assert np.allclose(t2_vals, 2.0)

def test_noisy_system_jax(noisy_system):
    set_backend('jax')
    param_vals = np.linspace(-0.5, 0.5, 100)

    t1_vals = [noisy_system.t1_effective() for _ in param_vals]
    t2_vals = [noisy_system.t2_effective() for _ in param_vals]

    assert np.allclose(t1_vals, 1.0)
    assert np.allclose(t2_vals, 2.0)

def test_backend_switch_between_numpy_and_jax(noisy_system):
    param_vals = np.linspace(-0.5, 0.5, 100)

    # Switch to numpy backend
    set_backend('numpy')
    t1_vals_numpy = [noisy_system.t1_effective() for _ in param_vals]
    t2_vals_numpy = [noisy_system.t2_effective() for _ in param_vals]

    # Switch to jax backend
    set_backend('jax')
    t1_vals_jax = [noisy_system.t1_effective() for _ in param_vals]
    t2_vals_jax = [noisy_system.t2_effective() for _ in param_vals]

    # Compare results
    assert np.allclose(t1_vals_numpy, t1_vals_jax)
    assert np.allclose(t2_vals_numpy, t2_vals_jax)

if __name__ == "__main__":
    pytest.main()
