import pytest
import numpy as np
import jax.numpy as jnp
from scqubits import backend_change
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.oscillator import Oscillator
from scqubits.core.transmon import Transmon
from scqubits.core.storage import SpectrumData
from scqubits.core.param_sweep import ParameterSweep
from scqubits.core.namedslots_array import NamedSlotsNdarray

@pytest.fixture
def setup_hilbert_space():
    osc1 = Oscillator(E_osc=5.0, l_osc=1.0, truncated_dim=10)
    osc2 = Oscillator(E_osc=4.0, l_osc=1.0, truncated_dim=10)
    transmon = Transmon(EJ=10.0, EC=0.2, ng=0.0, ncut=5, truncated_dim=5)
    hilbert_space = HilbertSpace([osc1, osc2, transmon])
    return hilbert_space

def dummy_update_hilbertspace(_):
    pass

@pytest.fixture
def setup_parameter_sweep(setup_hilbert_space):
    hilbert_space = setup_hilbert_space
    paramvals_by_name = {"param1": np.linspace(0, 1, 5)}
    parameter_sweep = ParameterSweep(
        hilbertspace=hilbert_space,
        paramvals_by_name=paramvals_by_name,
        update_hilbertspace=dummy_update_hilbertspace
    )
    return parameter_sweep

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_transitions(setup_parameter_sweep, backend):
    backend_change.set_backend(backend)
    parameter_sweep = setup_parameter_sweep

    # 设置测试参数
    subsystems = [parameter_sweep.hilbertspace[0], parameter_sweep.hilbertspace[1]]
    initial_state = (0, 0, 0)
    final_state = (1, 0, 0)

    transitions, energies = parameter_sweep.transitions(
        subsystems=subsystems,
        initial=initial_state,
        final=final_state,
        param_indices=(slice(None),)
    )

    # 检查返回值的类型
    assert isinstance(transitions, list)
    assert isinstance(energies, list)

    for transition in transitions:
        assert isinstance(transition, tuple)
        assert len(transition) == 2

    if backend == "numpy":
        for energy in energies:
            assert isinstance(energy, np.ndarray)
            energy = np.asarray(energy)  # 转换为普通的 numpy 数组
    elif backend == "jax":
        for energy in energies:
            assert isinstance(energy, jnp.ndarray)
            energy = jnp.asarray(energy)  # 转换为 JAX 数组

    # 检查转换数量和能量维度
    assert len(transitions) == len(energies)

    for energy in energies:
        assert energy.shape == (5,)

if __name__ == "__main__":
    pytest.main()
