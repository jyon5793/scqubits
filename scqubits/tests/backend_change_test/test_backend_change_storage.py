import pytest
import numpy as np
import jax.numpy as jnp
from hypothesis import given, settings, HealthCheck, strategies as st
from scqubits import backend_change
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.core.storage import SpectrumData

class DummyQubit(QubitBaseClass):
    def __init__(self, id_str=None):
        super().__init__(id_str)
        self.truncated_dim = 6
        self.dummy_param = 0.0

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
        return np.eye(evals_count, dtype=np.complex_)

    def get_spectrum_vs_paramvals(
        self, param_name, param_vals, evals_count=6, get_eigenstates=False, num_cpus=None
    ):
        eigenvalues = np.tile(np.arange(1, evals_count + 1), (len(param_vals), 1))
        eigenstates = np.tile(np.eye(evals_count), (len(param_vals), 1, 1))
        return SpectrumData(
            energy_table=eigenvalues,
            system_params={},
            param_name=param_name,
            param_vals=param_vals,
            state_table=eigenstates,
        )

    def hilbertdim(self):
        return 10


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_backend_consistency(backend):
    dummy_qubit = DummyQubit()
    backend_change.set_backend(backend)

    param_vals = np.linspace(0, 1, 5)
    operator = "some_operator"
    spectrum_data = dummy_qubit.get_matelements_vs_paramvals(operator, "dummy_param", param_vals)

    assert isinstance(spectrum_data, SpectrumData)
    assert spectrum_data.matrixelem_table.shape == (len(param_vals), dummy_qubit.truncated_dim, dummy_qubit.truncated_dim)
    assert np.all(spectrum_data.matrixelem_table == np.eye(dummy_qubit.truncated_dim, dtype=np.complex_))


@given(param_vals=st.lists(st.floats(0, 1), min_size=1, max_size=10))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_get_matelements_vs_paramvals(backend, param_vals):
    dummy_qubit = DummyQubit()
    backend_change.set_backend(backend)

    param_vals = np.array(param_vals)
    operator = "some_operator"
    spectrum_data = dummy_qubit.get_matelements_vs_paramvals(operator, "dummy_param", param_vals)

    assert isinstance(spectrum_data, SpectrumData)
    assert spectrum_data.matrixelem_table.shape == (len(param_vals), dummy_qubit.truncated_dim, dummy_qubit.truncated_dim)
    assert np.all(spectrum_data.matrixelem_table == np.eye(dummy_qubit.truncated_dim, dtype=np.complex_))


if __name__ == "__main__":
    pytest.main()
