import pytest
import numpy as np
from scqubits import ParameterSweep, HilbertSpace, backend_change
from scqubits.core.oscillator import Oscillator
from scqubits.core.transmon import Transmon
from scqubits.core.operators import number

def create_hilbert_space():
    osc1 = Oscillator(E_osc=5.0, l_osc=1.0, truncated_dim=10)
    osc2 = Oscillator(E_osc=6.0, l_osc=1.0, truncated_dim=10)
    transmon = Transmon(EJ=20.0, EC=0.2, ng=0.0, ncut=30, truncated_dim=10)
    return HilbertSpace([osc1, osc2, transmon])

def update_hilbert_space(sweep, osc_E, transmon_EJ):
    sweep.hilbertspace[0].E_osc = osc_E
    sweep.hilbertspace[2].EJ = transmon_EJ

def setup_parameters():
    paramvals_by_name = {
        'osc_E': np.linspace(4.0, 6.0, 5),
        'transmon_EJ': np.linspace(19.0, 21.0, 3)
    }
    return paramvals_by_name

def test_bare_spectrum_sweep_numpy():
    hilbert_space = create_hilbert_space()
    paramvals_by_name = setup_parameters()

    backend_change.set_backend('numpy')

    sweep = ParameterSweep(
        hilbertspace=hilbert_space,
        paramvals_by_name=paramvals_by_name,
        update_hilbertspace=update_hilbert_space,
        evals_count=5,
        autorun=False
    )

    evals, evecs = sweep._bare_spectrum_sweep()

    assert evals.shape == (3, 5, 3, 10)
    assert evecs.shape == (3, 5, 3, 10)

def test_bare_spectrum_sweep_jax():
    hilbert_space = create_hilbert_space()
    paramvals_by_name = setup_parameters()

    backend_change.set_backend('jax')

    sweep = ParameterSweep(
        hilbertspace=hilbert_space,
        paramvals_by_name=paramvals_by_name,
        update_hilbertspace=update_hilbert_space,
        evals_count=5,
        autorun=False
    )

    evals, evecs = sweep._bare_spectrum_sweep()

    assert evals.shape == (3, 5, 3, 10)
    assert evecs.shape == (3, 5, 3, 10)

def test_bare_spectrum_sweep_comparison():
    hilbert_space = create_hilbert_space()
    paramvals_by_name = setup_parameters()

    backend_change.set_backend('numpy')
    sweep_numpy = ParameterSweep(
        hilbertspace=hilbert_space,
        paramvals_by_name=paramvals_by_name,
        update_hilbertspace=update_hilbert_space,
        evals_count=5,
        autorun=False
    )
    evals_numpy, evecs_numpy = sweep_numpy._bare_spectrum_sweep()

    backend_change.set_backend('jax')
    sweep_jax = ParameterSweep(
        hilbertspace=hilbert_space,
        paramvals_by_name=paramvals_by_name,
        update_hilbertspace=update_hilbert_space,
        evals_count=5,
        autorun=False
    )
    evals_jax, evecs_jax = sweep_jax._bare_spectrum_sweep()

    np.testing.assert_allclose(evals_numpy, evals_jax, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(evecs_numpy, evecs_jax, rtol=1e-5, atol=1e-8)

def test_bare_spectrum_sweep_with_number_operator():
    hilbert_space = create_hilbert_space()
    paramvals_by_name = setup_parameters()

    backend_change.set_backend('numpy')
    sweep_numpy = ParameterSweep(
        hilbertspace=hilbert_space,
        paramvals_by_name=paramvals_by_name,
        update_hilbertspace=update_hilbert_space,
        evals_count=5,
        autorun=False
    )
    evals_numpy, evecs_numpy = sweep_numpy._bare_spectrum_sweep()

    backend_change.set_backend('jax')
    sweep_jax = ParameterSweep(
        hilbertspace=hilbert_space,
        paramvals_by_name=paramvals_by_name,
        update_hilbertspace=update_hilbert_space,
        evals_count=5,
        autorun=False
    )
    evals_jax, evecs_jax = sweep_jax._bare_spectrum_sweep()

    np.testing.assert_allclose(evals_numpy, evals_jax, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(evecs_numpy, evecs_jax, rtol=1e-5, atol=1e-8)

    num_op_numpy = number(10)
    num_op_jax = number(10)
    
    np.testing.assert_allclose(num_op_numpy, num_op_jax, rtol=1e-5, atol=1e-8)