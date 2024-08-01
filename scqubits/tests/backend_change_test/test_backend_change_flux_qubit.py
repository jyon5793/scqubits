import pytest
import numpy as np
from scqubits import FluxQubit
from scqubits import backend_change

def setup_flux_qubit():
    return FluxQubit(
        EJ1=1.0,
        EJ2=1.0,
        EJ3=0.8,
        ECJ1=0.016,
        ECJ2=0.016,
        ECJ3=0.021,
        ECg1=0.83,
        ECg2=0.83,
        ng1=0.0,
        ng2=0.0,
        flux=0.4,
        ncut=10
    )

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_n_operator(backend):
    backend_change.set_backend(backend)
    flux_qubit = setup_flux_qubit()
    
    # Calculate _n_operator with the current backend
    n_operator = flux_qubit._n_operator()

    # Switch backend
    other_backend = "jax" if backend == "numpy" else "numpy"
    backend_change.set_backend(other_backend)

    # Calculate _n_operator with the new backend
    other_n_operator = flux_qubit._n_operator()

    # Ensure both results are equal
    np.testing.assert_allclose(n_operator, other_n_operator, atol=1e-6)

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_kineticmat(backend):
    backend_change.set_backend(backend)
    flux_qubit = setup_flux_qubit()

    # Calculate kineticmat with the current backend
    kinetic_mat = flux_qubit.kineticmat()

    # Switch backend
    other_backend = "jax" if backend == "numpy" else "numpy"
    backend_change.set_backend(other_backend)

    # Calculate kineticmat with the new backend
    other_kinetic_mat = flux_qubit.kineticmat()

    # Ensure both results are equal
    np.testing.assert_allclose(kinetic_mat, other_kinetic_mat, atol=1e-6)
