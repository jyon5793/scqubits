import pytest
import numpy as np
import jax.numpy as jnp
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.extra.numpy import arrays
from sympy import symbols
from typing import List, Dict, Union
from scqubits import backend_change
from scqubits.core.storage import SpectrumData
from scqubits.core.symbolic_circuit import Node, Branch, SymbolicCircuit

@pytest.fixture
def create_dummy_qubit():
    class DummyQubit:
        def __init__(self):
            self.truncated_dim = 6
            self.dummy_param = 0.0

        def get_spectrum_vs_paramvals(self, param_name, param_vals, evals_count=6, get_eigenstates=False, num_cpus=None):
            eigenvalues = np.tile(np.arange(1, evals_count + 1), (len(param_vals), 1))
            eigenstates = np.tile(np.eye(evals_count), (len(param_vals), 1, 1))
            return SpectrumData(
                energy_table=eigenvalues,
                system_params={},
                param_name=param_name,
                param_vals=param_vals,
                state_table=eigenstates,
            )

        def get_matelements_vs_paramvals(self, operator, param_name, param_vals, evals_count=6, num_cpus=None):
            num_cpus = num_cpus or 1
            spectrumdata = self.get_spectrum_vs_paramvals(param_name, param_vals, evals_count=evals_count, get_eigenstates=True, num_cpus=num_cpus)
            paramvals_count = len(param_vals)
            matelem_table = backend_change.backend.empty(
                shape=(paramvals_count, evals_count, evals_count), dtype=backend_change.backend.complex_
            )
            paramval_before = getattr(self, param_name)
            for index, paramval in enumerate(param_vals):
                evecs = spectrumdata.state_table[index]
                setattr(self, param_name, paramval)
                if backend_change.backend.__name__ == "jax":
                    matelem_table = matelem_table.at[index].set(backend_change.backend.eye(evals_count, dtype=backend_change.backend.complex_))
                else:
                    matelem_table[index] = backend_change.backend.eye(evals_count, dtype=backend_change.backend.complex_)
            setattr(self, param_name, paramval_before)
            spectrumdata.matrixelem_table = matelem_table
            return spectrumdata

    return DummyQubit()

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_backend_consistency(create_dummy_qubit, backend):
    backend_change.set_backend(backend)
    dummy_qubit = create_dummy_qubit

    param_vals = np.linspace(0, 1, 5)
    operator = "some_operator"
    spectrum_data = dummy_qubit.get_matelements_vs_paramvals(operator, "dummy_param", param_vals)

    assert isinstance(spectrum_data, SpectrumData)
    assert spectrum_data.matrixelem_table.shape == (len(param_vals), dummy_qubit.truncated_dim, dummy_qubit.truncated_dim)
    assert np.all(spectrum_data.matrixelem_table == backend_change.backend.eye(dummy_qubit.truncated_dim, dtype=backend_change.backend.complex_))

@given(param_vals=arrays(dtype=float, shape=(5,), elements=st.floats(0, 1)))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_get_matelements_vs_paramvals(create_dummy_qubit, backend, param_vals):
    backend_change.set_backend(backend)
    dummy_qubit = create_dummy_qubit

    operator = "some_operator"
    spectrum_data = dummy_qubit.get_matelements_vs_paramvals(operator, "dummy_param", param_vals)

    assert isinstance(spectrum_data, SpectrumData)
    assert spectrum_data.matrixelem_table.shape == (len(param_vals), dummy_qubit.truncated_dim, dummy_qubit.truncated_dim)
    assert np.all(spectrum_data.matrixelem_table == backend_change.backend.eye(dummy_qubit.truncated_dim, dtype=backend_change.backend.complex_))

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_symbolic_circuit_generate_lagrangian(backend):
    backend_change.set_backend(backend)

    nodes = [Node(index=i, marker=0) for i in range(3)]
    branches = [
        Branch(nodes[0], nodes[1], "C", [1.0]),
        Branch(nodes[1], nodes[2], "L", [1.0]),
        Branch(nodes[2], nodes[0], "JJ", [1.0, 1.0])
    ]
    branch_var_dict = {symbols("EJ"): 1.0, symbols("EC"): 1.0}

    circuit = SymbolicCircuit(
        nodes_list=nodes,
        branches_list=branches,
        branch_var_dict=branch_var_dict,
        basis_completion="heuristic",
        use_dynamic_flux_grouping=False,
        initiate_sym_calc=True
    )

    lagrangian_θ, potential_θ, lagrangian_φ, potential_φ = circuit.generate_symbolic_lagrangian(substitute_params=True)

    assert lagrangian_θ is not None
    assert potential_θ is not None
    assert lagrangian_φ is not None
    assert potential_φ is not None

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_symbolic_circuit_generate_hamiltonian(backend):
    backend_change.set_backend(backend)

    nodes = [Node(index=i, marker=0) for i in range(3)]
    branches = [
        Branch(nodes[0], nodes[1], "C", [1.0]),
        Branch(nodes[1], nodes[2], "L", [1.0]),
        Branch(nodes[2], nodes[0], "JJ", [1.0, 1.0])
    ]
    branch_var_dict = {symbols("EJ"): 1.0, symbols("EC"): 1.0}

    circuit = SymbolicCircuit(
        nodes_list=nodes,
        branches_list=branches,
        branch_var_dict=branch_var_dict,
        basis_completion="heuristic",
        use_dynamic_flux_grouping=False,
        initiate_sym_calc=True
    )

    hamiltonian_symbolic = circuit.generate_symbolic_hamiltonian(substitute_params=True)

    assert hamiltonian_symbolic is not None

if __name__ == "__main__":
    pytest.main()
