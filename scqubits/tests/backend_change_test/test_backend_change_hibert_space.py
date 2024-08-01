import pytest
import numpy as np
import jax.numpy as jnp
from scqubits import backend_change
from scqubits.core.hilbert_space import HilbertSpace, InteractionTerm, InteractionTermStr
from scqubits.core.qubit_base import QubitBaseClass
from scqubits.core.oscillator import Oscillator
from qutip import Qobj

class DummySubsystem(QubitBaseClass):
    def __init__(self, id_str, truncated_dim):
        self._id_str = id_str
        self.truncated_dim = truncated_dim

    def eigenvals(self, evals_count):
        return np.arange(evals_count)

    def eigensys(self, evals_count):
        return np.arange(evals_count), np.eye(evals_count)

    def hamiltonian(self):
        return np.eye(self.truncated_dim)

    @property
    def default_params(self):
        return {}

    @property
    def hilbertdim(self):
        return self.truncated_dim

class DummyOscillator(Oscillator):
    def __init__(self, id_str, truncated_dim):
        self._id_str = id_str
        self.truncated_dim = truncated_dim

    def eigenvals(self, evals_count):
        return np.arange(evals_count)

    def eigensys(self, evals_count):
        return np.arange(evals_count), np.eye(evals_count)

    def hamiltonian(self):
        return np.eye(self.truncated_dim)

    @property
    def default_params(self):
        return {}

    @property
    def hilbertdim(self):
        return self.truncated_dim

@pytest.fixture
def hilbert_space():
    subsystems = [
        DummySubsystem("qubit1", 2),
        DummyOscillator("oscillator1", 5)
    ]
    interaction_terms = [
        InteractionTerm(g_strength=0.1, operator_list=[(0, np.eye(2)), (1, np.eye(5))])
    ]
    return HilbertSpace(subsystem_list=subsystems, interaction_list=interaction_terms)

def test_bare_hamiltonian_numpy(hilbert_space):
    backend_change.set_backend('numpy')
    hamiltonian = hilbert_space.bare_hamiltonian()
    assert isinstance(hamiltonian, Qobj)
    assert hamiltonian.shape == (10, 10)

def test_bare_hamiltonian_jax(hilbert_space):
    backend_change.set_backend('jax')
    hamiltonian = hilbert_space.bare_hamiltonian()
    assert isinstance(hamiltonian, Qobj)
    assert hamiltonian.shape == (10, 10)

def test_interaction_hamiltonian_numpy(hilbert_space):
    backend_change.set_backend('numpy')
    hamiltonian = hilbert_space.interaction_hamiltonian()
    assert isinstance(hamiltonian, Qobj)
    assert hamiltonian.shape == (10, 10)

def test_interaction_hamiltonian_jax(hilbert_space):
    backend_change.set_backend('jax')
    hamiltonian = hilbert_space.interaction_hamiltonian()
    assert isinstance(hamiltonian, Qobj)
    assert hamiltonian.shape == (10, 10)

def test_dimension_numpy(hilbert_space):
    backend_change.set_backend('numpy')
    assert hilbert_space.dimension == 10

def test_dimension_jax(hilbert_space):
    backend_change.set_backend('jax')
    assert hilbert_space.dimension == 10

def test_generate_lookup_numpy(hilbert_space):
    backend_change.set_backend('numpy')
    hilbert_space.generate_lookup()
    assert 'evals' in hilbert_space._data
    assert 'evecs' in hilbert_space._data

def test_generate_lookup_jax(hilbert_space):
    backend_change.set_backend('jax')
    hilbert_space.generate_lookup()
    assert 'evals' in hilbert_space._data
    assert 'evecs' in hilbert_space._data

def test_eigenvals_numpy(hilbert_space):
    backend_change.set_backend('numpy')
    evals = hilbert_space.eigenvals(evals_count=5)
    assert len(evals) == 5

def test_eigenvals_jax(hilbert_space):
    backend_change.set_backend('jax')
    evals = hilbert_space.eigenvals(evals_count=5)
    assert len(evals) == 5

def test_eigensys_numpy(hilbert_space):
    backend_change.set_backend('numpy')
    evals, evecs = hilbert_space.eigensys(evals_count=2)
    assert len(evals) == 2
    assert len(evecs) == 2  # 确保返回两个特征向量
    for evec in evecs:
        evec_np = evec.full().flatten()  # 将Qobj特征向量转换为numpy数组
        assert evec_np.shape == (hilbert_space.dimension,)

def test_eigensys_jax(hilbert_space):
    backend_change.set_backend('jax')
    evals, evecs = hilbert_space.eigensys(evals_count=2)
    assert len(evals) == 2
    assert len(evecs) == 2
    for evec in evecs:
        evec_np = evec.full().flatten()
        assert evec_np.shape == (hilbert_space.dimension,)

if __name__ == '__main__':
    pytest.main()
