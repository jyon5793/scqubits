import pytest
import numpy as np
from scqubits import backend_change
from scqubits.core.qubit_base import QubitBaseClass, QuantumSystem
from typing import Union, Tuple, Dict
from numpy import ndarray
from scqubits.core.storage import SpectrumData

class DummyQubit(QubitBaseClass):
    """Dummy Qubit class for testing purposes"""
    def __init__(self, id_str=None, evals_method=None, evals_method_options=None, esys_method=None, esys_method_options=None):
        super().__init__(id_str, evals_method, evals_method_options, esys_method, esys_method_options)
        self.truncated_dim = 10
        self._init_params = ['id_str']

    def hamiltonian(self):
        """Dummy Hamiltonian for testing"""
        return np.diag(np.arange(1, self.truncated_dim + 1))

    def hilbertdim(self) -> int:
        """Return dimension of Hilbert space"""
        return self.truncated_dim

    @staticmethod
    def default_params():
        return {
            'id_str': 'dummy'
        }

@pytest.fixture
def dummy_qubit():
    return DummyQubit()

def test_backend_consistency(dummy_qubit):
    # Test eigenvals method
    backend_change.set_backend('numpy')
    numpy_evals = dummy_qubit.eigenvals(evals_count=dummy_qubit.truncated_dim)
    
    backend_change.set_backend('jax')
    jax_evals = dummy_qubit.eigenvals(evals_count=dummy_qubit.truncated_dim)

    assert np.allclose(numpy_evals, jax_evals)

    # Test eigensys method
    backend_change.set_backend('numpy')
    numpy_evals, numpy_evecs = dummy_qubit.eigensys(evals_count=dummy_qubit.truncated_dim)
    
    backend_change.set_backend('jax')
    jax_evals, jax_evecs = dummy_qubit.eigensys(evals_count=dummy_qubit.truncated_dim)

    assert np.allclose(numpy_evals, jax_evals)
    assert np.allclose(numpy_evecs, jax_evecs)

@pytest.fixture(autouse=True)
def reset_backend():
    # Reset the backend to numpy after each test
    yield
    backend_change.set_backend('numpy')

class DummyQubit(QubitBaseClass):
    def __init__(self, id_str=None):
        super().__init__(id_str)
        self.truncated_dim = 6  # Adjust to 6 for testing
        self.dummy_param = 0.0  # Add default attribute

    def hamiltonian(self):
        return np.diag(np.arange(1, 11))

    @staticmethod
    def default_params():
        return {}

    def potential(self, phi):
        return phi ** 2

    def wavefunction(self, esys, which=0, phi_grid=None):
        pass

    def matrixelement_table(self, operator, evecs=None, evals_count=6):
        # For testing, return a fixed matrix
        return np.eye(evals_count, dtype=np.complex_)

    def get_spectrum_vs_paramvals(
        self, param_name, param_vals, evals_count=6, get_eigenstates=False, num_cpus=None
    ):
        # For testing, return a SpectrumData object with fixed eigenvalues and eigenstates
        eigenvalues = np.tile(np.arange(1, evals_count + 1), (len(param_vals), 1))
        eigenstates = np.tile(np.eye(evals_count), (len(param_vals), 1, 1))
        return SpectrumData(
            energy_table=eigenvalues,
            state_table=eigenstates,
            system_params={},
            param_name=param_name,
            param_vals=param_vals,
        )

    def hilbertdim(self):
        return 10  # or any appropriate integer value


@pytest.fixture
def dummy_qubit():
    return DummyQubit()


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_get_matelements_vs_paramvals(dummy_qubit, backend):
    backend_change.set_backend(backend)

    param_vals = np.linspace(0, 1, 5)
    operator = "some_operator"
    spectrum_data = dummy_qubit.get_matelements_vs_paramvals(operator, "dummy_param", param_vals)

    assert isinstance(spectrum_data, SpectrumData)
    assert spectrum_data.matrixelem_table.shape == (len(param_vals), dummy_qubit.truncated_dim, dummy_qubit.truncated_dim)
    assert np.all(spectrum_data.matrixelem_table == np.eye(dummy_qubit.truncated_dim, dtype=np.complex_))

if __name__ == '__main__':
    pytest.main()
