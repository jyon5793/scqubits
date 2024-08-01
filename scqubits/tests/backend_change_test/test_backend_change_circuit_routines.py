import pytest
import sympy as sm
from scqubits import backend_change as backend
from scqubits import Circuit
from scqubits.core.circuit_routines import CircuitRoutines
import numpy as np

class TestCircuitRoutines:

    @pytest.fixture
    def circuit_routines(self):
        # 初始化 CircuitRoutines 实例
        yaml_str = """
        branches:
        - [JJ, 1, 2, 10]
        - [L, 1, 2, 0.1]
        """
        return Circuit(yaml_str, from_file=False, ext_basis="harmonic")

    def test_serialize_deserialize(self, circuit_routines):
        serialized = circuit_routines.serialize()
        deserialized = CircuitRoutines.deserialize(serialized)
        assert isinstance(deserialized, CircuitRoutines)

    # def test_return_root_child(self, circuit_routines):
    #     result = circuit_routines.return_root_child(1)
    #     assert result == circuit_routines

    # def test_return_parent_circuit(self, circuit_routines):
    #     parent = circuit_routines.return_parent_circuit()
    #     assert parent == circuit_routines

    # def test_is_expression_purely_harmonic(self):
    #     expr = sm.sympify("Q1**2 + θ1**2")
    #     result = CircuitRoutines._is_expression_purely_harmonic(expr)
    #     assert result == True

    # def test_diagonalize_purely_harmonic_hamiltonian(self, circuit_routines):
    #     circuit_routines.is_purely_harmonic = True
    #     circuit_routines.hamiltonian_symbolic = sm.sympify("Q1**2 + θ1**2")
    #     result = circuit_routines._diagonalize_purely_harmonic_hamiltonian()
    #     assert result is not None

    # def test_transform_hamiltonian(self, circuit_routines):
    #     hamiltonian = sm.sympify("Q1**2 + θ1**2")
    #     transformation_matrix = sm.eye(2)
    #     result = circuit_routines._transform_hamiltonian(hamiltonian, transformation_matrix)
    #     assert result is not None

    # def test_offset_charge_transformation(self, circuit_routines):
    #     with pytest.raises(Exception):
    #         circuit_routines.offset_charge_transformation()

    # def test_setattr(self, circuit_routines):
    #     with pytest.raises(Exception):
    #         circuit_routines._frozen = True
    #         circuit_routines.ext_basis = "discretized"

    # def test_reduce_setstate(self, circuit_routines):
    #     pickled_data = circuit_routines.__reduce__()
    #     restored_instance = CircuitRoutines.__new__(CircuitRoutines)
    #     restored_instance.__setstate__(pickled_data[2])
    #     assert restored_instance is not None

    # def test_default_params(self):
    #     params = CircuitRoutines.default_params()
    #     assert isinstance(params, dict)

    # def test_cutoffs_dict(self, circuit_routines):
    #     cutoffs = circuit_routines.cutoffs_dict()
    #     assert isinstance(cutoffs, dict)

    # def test_set_property_and_update_param_vars(self, circuit_routines):
    #     circuit_routines._set_property_and_update_param_vars("EJ", 20.0)
    #     assert circuit_routines._EJ == 20.0

    # def test_set_property_and_update_ext_flux_or_charge(self, circuit_routines):
    #     circuit_routines._set_property_and_update_ext_flux_or_charge("ng", 0.1)
    #     assert circuit_routines._ng == 0.1

    # def test_set_property_and_update_cutoffs(self, circuit_routines):
    #     circuit_routines._set_property_and_update_cutoffs("ncut", 40)
    #     assert circuit_routines._ncut == 40

    # def test_set_property_and_update_user_changed_parameter(self, circuit_routines):
    #     circuit_routines._set_property_and_update_user_changed_parameter("user_changed", "param")
    #     assert circuit_routines._user_changed == "param"

    # def test_make_property(self, circuit_routines):
    #     circuit_routines._make_property("test_property", 10, "update_param_vars")
    #     assert circuit_routines.test_property == 10

    # def test_set_discretized_phi_range(self, circuit_routines):
    #     circuit_routines.set_discretized_phi_range((1,), (0, 2*np.pi))
    #     assert circuit_routines.discretized_phi_range[1] == (0, 2*np.pi)

    # def test_set_and_return(self, circuit_routines):
    #     result = circuit_routines.set_and_return("ext_basis", "discretized")
    #     assert result.ext_basis == "discretized"

    # def test_get_ext_basis(self, circuit_routines):
    #     basis = circuit_routines.get_ext_basis()
    #     assert basis == "harmonic"

    # def test_sync_parameters_with_parent(self, circuit_routines):
    #     with pytest.raises(Exception):
    #         circuit_routines.sync_parameters_with_parent()

    # def test_set_sync_status_to_True(self, circuit_routines):
    #     circuit_routines._set_sync_status_to_True()
    #     assert circuit_routines._out_of_sync == False

    # def test_receive(self, circuit_routines):
    #     with pytest.raises(Exception):
    #         circuit_routines.receive("event", circuit_routines)

    # def test_store_updated_subsystem_index(self, circuit_routines):
    #     with pytest.raises(Exception):
    #         circuit_routines._store_updated_subsystem_index(0)

    # def test_fetch_symbolic_hamiltonian(self, circuit_routines):
    #     hamiltonian = circuit_routines.fetch_symbolic_hamiltonian()
    #     assert hamiltonian is not None

    # def test_update(self, circuit_routines):
    #     circuit_routines.update()
    #     assert circuit_routines._frozen == True

    # def test_perform_internal_updates(self, circuit_routines):
    #     circuit_routines._perform_internal_updates()
    #     assert circuit_routines._user_changed_parameter == False

    # def test_update_bare_esys(self, circuit_routines):
    #     with pytest.raises(Exception):
    #         circuit_routines._update_bare_esys()

    # def test_discretized_grids_dict_for_vars(self, circuit_routines):
    #     grids = circuit_routines.discretized_grids_dict_for_vars()
    #     assert isinstance(grids, dict)

    # def test_constants_in_subsys(self, circuit_routines):
    #     h_sys = sm.sympify("Q1**2 + θ1**2")
    #     constants_expr = sm.sympify("EC + EL")
    #     result = circuit_routines._constants_in_subsys(h_sys, constants_expr)
    #     assert result is not None

    # def test_list_of_constants_from_expr(self, circuit_routines):
    #     expr = sm.sympify("EC + EL")
    #     result = circuit_routines._list_of_constants_from_expr(expr)
    #     assert isinstance(result, list)

    # def test_check_truncation_indices(self, circuit_routines):
    #     with pytest.raises(Exception):
    #         circuit_routines._check_truncation_indices()

    # def test_sym_hamiltonian_for_var_indices(self, circuit_routines):
    #     hamiltonian_expr = sm.sympify("Q1**2 + θ1**2")
    #     subsys_index_list = [1]
    #     result = circuit_routines._sym_hamiltonian_for_var_indices(hamiltonian_expr, subsys_index_list)
    #     assert result is not None

    # def test_generate_subsystems(self, circuit_routines):
    #     circuit_routines.generate_subsystems()
    #     assert isinstance(circuit_routines.subsystems, list)

    # def test_get_eigenstates(self, circuit_routines):
    #     eigenstates = circuit_routines.get_eigenstates()
    #     assert isinstance(eigenstates, np.ndarray)

    # def test_get_subsystem_index(self, circuit_routines):
    #     index = circuit_routines.get_subsystem_index(1)
    #     assert index == 0

    # def test_update_interactions(self, circuit_routines):
    #     circuit_routines.update_interactions()
    #     assert circuit_routines.hilbert_space.interaction_list == []

    # def test_evaluate_symbolic_expr(self, circuit_routines):
    #     sym_expr = sm.sympify("Q1**2 + θ1**2")
    #     result = circuit_routines._evaluate_symbolic_expr(sym_expr)
    #     assert isinstance(result, qt.Qobj)

    # def test_operator_from_sym_expr_wrapper(self, circuit_routines):
    #     sym_expr = sm.sympify("Q1**2 + θ1**2")
    #     result = circuit_routines._operator_from_sym_expr_wrapper(sym_expr)
    #     assert callable(result)

    # def test_generate_symbols_list(self, circuit_routines):
    #     result = circuit_routines._generate_symbols_list("Q", [1, 2, 3])
    #     assert isinstance(result, list)

    # def test_set_vars(self, circuit_routines):
    #     circuit_routines._set_vars()
    #     assert isinstance(circuit_routines.vars, dict)

    # def test_set_vars_no_hd(self, circuit_routines):
    #     circuit_routines._set_vars_no_hd()
    #     assert isinstance(circuit_routines.vars, dict)

    # def test_shift_harmonic_oscillator_potential(self, circuit_routines):
    #     hamiltonian = sm.sympify("Q1**2 + θ1**2")
    #     result = circuit_routines._shift_harmonic_oscillator_potential(hamiltonian)
    #     assert result is not None

    # def test_generate_sym_potential(self, circuit_routines):
    #     potential = circuit_routines.generate_sym_potential()
    #     assert potential is not None

    # def test_generate_hamiltonian_sym_for_numerics(self, circuit_routines):
    #     result = circuit_routines.generate_hamiltonian_sym_for_numerics()
    #     assert result is not None

    # def test_get_cutoffs(self, circuit_routines):
    #     cutoffs = circuit_routines.get_cutoffs()
    #     assert isinstance(cutoffs, dict)

    # def test_collect_cutoff_values(self, circuit_routines):
    #     values = list(circuit_routines._collect_cutoff_values())
    #     assert isinstance(values, list)

    # def test_hilbertdim(self, circuit_routines):
    #     dim = circuit_routines.hilbertdim()
    #     assert isinstance(dim, int)

    # def test_kron_operator(self, circuit_routines):
    #     operator = sm.eye(2)
    #     result = circuit_routines._kron_operator(operator, 1)
    #     assert result is not None

    # def test_sparsity_adaptive(self, circuit_routines):
    #     matrix = sm.eye(2)
    #     result = circuit_routines._sparsity_adaptive(matrix)
    #     assert result is not None

    # def test_identity_qobj(self, circuit_routines):
    #     result = circuit_routines._identity_qobj()
    #     assert isinstance(result, qt.Qobj)

    # def test_identity(self, circuit_routines):
    #     result = circuit_routines._identity()
    #     assert result is not None

    # def test_exp_i_operator(self, circuit_routines):
    #     var_sym = sm.symbols("θ1")
    #     result = circuit_routines.exp_i_operator(var_sym, 1)
    #     assert result is not None

    # def test_evaluate_matrix_sawtooth_terms(self, circuit_routines):
    #     saw_expr = sm.Function("saw")(sm.symbols("θ1"))
    #     result = circuit_routines._evaluate_matrix_sawtooth_terms(saw_expr)
    #     assert isinstance(result, qt.Qobj)

    # def test_evaluate_matrix_cosine_terms(self, circuit_routines):
    #     junction_potential = sm.sympify("cos(θ1)")
    #     result = circuit_routines._evaluate_matrix_cosine_terms(junction_potential)
    #     assert isinstance(result, qt.Qobj)

    # def test_set_harmonic_basis_osc_params(self, circuit_routines):
    #     circuit_routines._set_harmonic_basis_osc_params()
    #     assert isinstance(circuit_routines.osc_lengths, dict)

    # def test_wrapper_operator_for_purely_harmonic_system(self, circuit_routines):
    #     result = circuit_routines._wrapper_operator_for_purely_harmonic_system("θ1")
    #     assert callable(result)

    # def test_generate_operator_methods(self, circuit_routines):
    #     result = circuit_routines._generate_operator_methods()
    #     assert isinstance(result, dict)

    # def test_get_params(self, circuit_routines):
    #     params = circuit_routines.get_params()
    #     assert isinstance(params, list)

    # def test_offset_free_charge_values(self, circuit_routines):
    #     values = circuit_routines.offset_free_charge_values()
    #     assert isinstance(values, list)

    # def test_set_operators(self, circuit_routines):
    #     result = circuit_routines.set_operators()
    #     assert isinstance(result, dict)

if __name__ == "__main__":
    pytest.main()
