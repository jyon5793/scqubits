# spectrum_utils.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
#    This source code is licensed under the BSD-style license found in the
#    LICENSE file in the root directory of this source tree.
############################################################################

import cmath

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import numpy as np
import qutip as qt
import scipy as sp

from numpy import ndarray
from qutip import Qobj
from scipy.sparse import csc_matrix, dia_matrix, csr_matrix
from jax import custom_vjp

import scqubits.settings as settings

if TYPE_CHECKING:
    from scqubits import Oscillator, ParameterSweep, SpectrumData
    from scqubits.core.qubit_base import QubitBaseClass
    from scqubits.io_utils.fileio_qutip import QutipEigenstates

from scqubits.utils.typedefs import QuantumSys
from scqubits.utils.misc import Qobj_to_scipy_csc_matrix
from scqubits import backend_change as bc


def eigsh_safe(*args, **kwargs):
    from scqubits.backend_change import backend_dependent_vjp

    @backend_dependent_vjp
    def eigsh_safe_impl(*args, **kwargs):
        mat_size = args[0].shape[0]
        kwargs["v0"] = settings.RANDOM_ARRAY[:mat_size]
        return_eigenvectors = kwargs.get("return_eigenvectors", True)
    
        if return_eigenvectors:
            evals, evecs = sp.sparse.linalg.eigsh(*args, **kwargs)
            if has_degeneracy(evals):
                evecs, _ = sp.linalg.qr(evecs, mode="economic")
            return evals, evecs
        else:
            evals = sp.sparse.linalg.eigsh(*args, **kwargs)
            return evals

    return eigsh_safe_impl(*args, **kwargs)

# 前向计算
def eigsh_safe_fwd(A, k=6, return_eigenvectors=True, **kwargs):
    evals, evecs = eigsh_safe(A, k=k, return_eigenvectors=return_eigenvectors, **kwargs)
    return (evals, evecs), (A, evals, evecs)

# 反向传播
def eigsh_safe_bwd(residuals, g):
    A, evals, evecs = residuals
    grad_evals, grad_evecs = g
    
    grad_A = bc.backend.zeros_like(A)
    
    # 实现梯度的计算逻辑
    for i in range(len(evals)):
        for j in range(len(evals)):
            if i != j:
                grad_A += (
                    grad_evecs[:, i].dot(evecs[:, j])
                    / (evals[i] - evals[j])
                    * (bc.backend.outer(evecs[:, j], evecs[:, i]) + bc.backend.outer(evecs[:, i], evecs[:, j]))
                )
    
    return (grad_A,)
eigsh_safe = bc.backend.bind_custom_vjp(eigsh_safe_fwd, eigsh_safe_bwd,eigsh_safe)


def has_degeneracy(evals: bc.backendndarray) -> bool:
    evals_rightpad = bc.backend.pad(evals, (0, 1))
    evals_leftpad = bc.backend.pad(evals, (1, 0))
    evals_neighbor_diffs = evals_leftpad - evals_rightpad
    return bc.backend.isclose(bc.backend.min(bc.backend.abs(evals_neighbor_diffs)), 0)


def order_eigensystem(
    evals: bc.backend.bc.backendndarray, evecs: bc.backend.ndarray
) -> Tuple[bc.backend.ndarray, bc.backend.ndarray]:
    """Takes eigenvalues and corresponding eigenvectors and orders them (in place)
    according to the eigenvalues (from smallest to largest; real valued eigenvalues
    are assumed). Compare http://stackoverflow.com/questions/22806398.

    Parameters
    ----------
    evals:
        array of eigenvalues
    evecs:
        array containing eigenvectors; evecs[:, 0] is the first eigenvector etc.
    """
    from scqubits.backend_change import backend_dependent_vjp
    @backend_dependent_vjp
    def order_eigensystem_impl(evals: bc.backend.ndarray, evecs: bc.backend.ndarray)-> Tuple[bc.backend.ndarray, bc.backend.ndarray]:
        ordered_evals_indices = evals.argsort()  # sort manually
        evals[:] = evals[ordered_evals_indices]
        evecs[:] = evecs[:, ordered_evals_indices]
        return evals, evecs
    return order_eigensystem_impl(evals,evecs)

# 前向计算
def order_eigensystem_fwd(evals, evecs):
    ordered_evals_indices = evals.argsort()
    sorted_evals = evals[ordered_evals_indices]
    sorted_evecs = evecs[:, ordered_evals_indices]
    return (sorted_evals, sorted_evecs), ordered_evals_indices

# 反向传播
def order_eigensystem_bwd(ordered_evals_indices, g):
    grad_evals, grad_evecs = g
    
    grad_evals_out = bc.backend.zeros_like(grad_evals)
    grad_evecs_out = bc.backend.zeros_like(grad_evecs)
    
    grad_evals_out = grad_evals[ordered_evals_indices.argsort()]
    grad_evecs_out = grad_evecs[:, ordered_evals_indices.argsort()]
    
    return grad_evals_out, grad_evecs_out
order_eigensystem = bc.backend.bind_custom_vjp(order_eigensystem_fwd, order_eigensystem_bwd, order_eigensystem)


def extract_phase(
    complex_array: bc.backend.ndarray, position: Optional[Tuple[bc.backend.int_, ...]] = None
) -> bc.backend.float_:
    """Extracts global phase from `complex_array` at given `position`. If position is
    not specified, the `position` is set as follows. Find the maximum between the
    leftmost point and the halfway point of the wavefunction. The position of that
    point is used to determine the phase factor to be eliminated.

    Parameters
    ----------
    complex_array:
        complex-valued array
    position:
        position where the phase is extracted (default value = None)
    """
    from scqubits.backend_change import backend_dependent_vjp
    @backend_dependent_vjp
    def extract_phase_impl(complex_array: bc.backend.ndarray, position: Optional[Tuple[bc.backend.int_, ...]] = None) -> bc.backend.float_:
        if position is None:
            halfway_position = (complex_array.shape[0]) // 2
            flattened_position = bc.backend.argmax(
                bc.backend.abs(complex_array[:halfway_position])
            )  # extract phase from element with largest amplitude modulus
            position = bc.backend.unravel_index(flattened_position, complex_array.shape)
        return cmath.phase(complex_array[position])
    return extract_phase_impl(complex_array,position)

# 前向计算
def extract_phase_fwd(complex_array, position=None):
    phase = extract_phase(complex_array, position)
    return phase, (complex_array, position, phase)

# 反向传播
def extract_phase_bwd(residuals, g):
    complex_array, position, phase = residuals
    """
    由于相位在 −π 和 π 之间是环绕的，这意味着在某些点上，输入的微小变化可能导致输出相位发生大幅度跳跃。这种跳跃导致相位在这些点上不可微
    """
    grad_array = bc.backend.zeros_like(complex_array)
    return (grad_array,)

# 绑定前向和反向传播
extract_phase= bc.backend.bind_custom_vjp(extract_phase_fwd, extract_phase_bwd,extract_phase)


def standardize_phases(complex_array: bc.backend.ndarray) -> bc.backend.ndarray:
    """Uses `extract_phase` to obtain global phase from `array` and returns
    standardized array with global phase factor standardized.

    Parameters
    ----------
    complex_array:
        complex
    """
    phase = extract_phase(complex_array)
    std_array = complex_array * bc.backend.exp(-1j * phase)
    return std_array


def standardize_sign(real_array: bc.backend.ndarray) -> bc.backend.ndarray:
    """Standardizes the sign of a real-valued wavefunction by calculating the sign of
    the sum of all amplitudes up to the wavefunctions mid-position and making it
    positive.

    Summing up to the midpoint only is to address the danger that the sum is
    actually zero, which may be the case for odd wavefunctions taken over an interval
    centered at zero.
    """
    halfway_position = len(real_array) // 2
    return bc.backend.sign(bc.backend.sum(real_array[:halfway_position])) * real_array


# -Matrix elements and operators (outside qutip) --------------------------------------


def matrix_element(
    state1: Union[bc.backend.ndarray, qt.Qobj],
    operator: Union[bc.backend.ndarray, bc.backend.csc_matrix, qt.Qobj],
    state2: Union[bc.backend.ndarray, qt.Qobj],
) -> Union[bc.backend.float_, bc.backend.complex_]:
    """Calculate the matrix element `<state1|operator|state2>`.

    Parameters
    ----------
    state1:
        state vector/ket
    state2:
        state vector/ket
    operator:
        representation of an operator

    Returns
    -------
        matrix element
    """
    if isinstance(operator, qt.Qobj):
        op_matrix = operator.data
    else:
        op_matrix = operator

    if isinstance(state1, qt.Qobj):
        vec1 = Qobj_to_scipy_csc_matrix(state1)
    else:
        vec1 = state1

    if isinstance(state2, qt.Qobj):
        vec2 = Qobj_to_scipy_csc_matrix(state2)
    else:
        vec2 = state2

    return vec1.conj().T @ op_matrix @ vec2


def get_matrixelement_table(
    operator: Union[bc.backend.ndarray, bc.backend.csc_matrix, bc.backend.dia_matrix, qt.Qobj],
    state_table: Union[bc.backend.ndarray, qt.Qobj],
) -> bc.backend.ndarray:
    """Calculates a table of matrix elements.

    Parameters
    ----------
    operator:
        operator with respect to which matrix elements are to be calculated
    state_table:
        list or array of numpy arrays representing the states `|v0>, |v1>, ...`
        Note: `state_table` is expected to be in scipy's `eigsh` transposed form.

    Returns
    -------
        table of matrix elements
    """
    if isinstance(operator, qt.Qobj):
        states_in_columns = state_table.T
    else:
        states_in_columns = state_table

    mtable = states_in_columns.conj().T @ operator @ states_in_columns
    return mtable


def closest_dressed_energy(
    bare_energy: bc.backend.float_, dressed_energy_vals: bc.backend.ndarray
) -> bc.backend.float_:
    """For a given bare energy value, this returns the closest lying dressed energy
    value from an array.

    Parameters
    ----------
    bare_energy:
        bare energy value
    dressed_energy_vals:
        array of dressed-energy values

    Returns
    -------
        element from `dressed_energy_vals` closest to `bare_energy`
    """
    from scqubits.backend_change import backend_dependent_vjp
    @backend_dependent_vjp
    def closest_dressed_energy_impl(bare_energy, dressed_energy_vals):
        index = (bc.backend.abs(dressed_energy_vals - bare_energy)).argmin()
        return dressed_energy_vals[index]
    return closest_dressed_energy_impl(bare_energy, dressed_energy_vals)

def closest_dressed_energy_fwd(bare_energy, dressed_energy_vals):
    closest_energy = closest_dressed_energy(bare_energy, dressed_energy_vals)
    return closest_energy, (bare_energy, dressed_energy_vals, closest_energy)

# 反向传播
def closest_dressed_energy_bwd(residuals, g):
    bare_energy, dressed_energy_vals, closest_energy = residuals
    
    # 计算梯度
    grad_bare_energy = 0.0
    grad_dressed_energy_vals = bc.backend.zeros_like(dressed_energy_vals)
    
    index = bc.backend.argmin(bc.backend.abs(dressed_energy_vals - bare_energy))
    
    # 传播梯度
    grad_dressed_energy_vals = grad_dressed_energy_vals.at[index].set(g)
    
    return grad_bare_energy, grad_dressed_energy_vals

closest_dressed_energy = bc.backend.bind_custom_vjp(closest_dressed_energy_fwd, closest_dressed_energy_bwd,closest_dressed_energy)

def get_eigenstate_index_maxoverlap(
    eigenstates_qobj: "QutipEigenstates",
    reference_state_qobj: qt.Qobj,
    return_overlap: bool = False,
) -> Union[bc.backend.int_, Tuple[bc.backend.int_, bc.backend.float_], None]:
    """For given list of qutip states, find index of the state that has largest
    overlap with the qutip ket `reference_state_qobj`. If `|overlap|` is smaller than
    0.5, return None.

    Parameters
    ----------
    eigenstates_qobj:
        as obtained from qutip `.eigenstates()`
    reference_state_qobj:
        specific reference state
    return_overlap:
        set to true if the value of largest overlap should be also returned
        (default value = False)

    Returns
    -------
        index of eigenstate from `eigenstates_Qobj` with the largest overlap with the
        `reference_state_qobj`, None if `|overlap|<0.5`
    """
    overlaps = bc.backend.asarray(
        [
            eigenstates_qobj[j].overlap(reference_state_qobj)
            for j in range(len(eigenstates_qobj))
        ]
    )
    max_overlap = bc.backend.max(bc.backend.abs(overlaps))
    if max_overlap < 0.5:
        return None
    index = (bc.backend.abs(overlaps)).argmax()
    if return_overlap:
        return index, bc.backend.abs(overlaps[index])
    return index


def absorption_spectrum(spectrum_data: "SpectrumData") -> "SpectrumData":
    """Takes spectral data of energy eigenvalues and returns the absorption spectrum
    relative to a state of given index. Calculated by subtracting from eigenenergies
    the energy of the select state. Resulting negative frequencies, if the reference
    state is not the ground state, are omitted.
    """
    assert isinstance(spectrum_data.energy_table, ndarray)
    spectrum_data.energy_table = spectrum_data.energy_table.clip(min=0.0)  # type:ignore
    return spectrum_data


def emission_spectrum(spectrum_data: "SpectrumData") -> "SpectrumData":
    """Takes spectral data of energy eigenvalues and returns the emission spectrum
    relative to a state of given index. The resulting "upwards" transition
    frequencies are calculated by subtracting from eigenenergies the energy of the
    select state, and multiplying the result by -1. Resulting negative frequencies,
    corresponding to absorption instead, are omitted.
    """
    assert isinstance(spectrum_data.energy_table, bc.backend.ndarray)
    spectrum_data.energy_table *= -1.0
    spectrum_data.energy_table = spectrum_data.energy_table.clip(min=0.0)  # type:ignore
    return spectrum_data


def convert_evecs_to_ndarray(evecs_qutip: bc.backend.ndarray) -> bc.backend.ndarray:
    """Takes a qutip eigenstates array, as obtained with .eigenstates(), and converts
    it into a pure numpy array.

    Parameters
    ----------
    evecs_qutip:
        ndarray of eigenstates in qt.Qobj format

    Returns
    -------
        converted eigenstate data
    """
    evals_count = len(evecs_qutip)
    dimension = evecs_qutip[0].shape[0]
    evecs_ndarray = bc.backend.empty((evals_count, dimension), dtype=bc.backend.complex_)
    for index, eigenstate in enumerate(evecs_qutip):
        # evecs_ndarray[index] = eigenstate.full()[:, 0]
        evecs_ndarray = bc.backend.array_solve(evecs_ndarray, index, eigenstate.full()[:, 0])
    return evecs_ndarray


def convert_matrix_to_qobj(
    operator: Union[bc.backend.ndarray, bc.backend.csc_matrix, bc.backend.dia_matrix],
    subsystem: Union["QubitBaseClass", "Oscillator"],
    op_in_eigenbasis: bool,
    evecs: Optional[bc.backend.ndarray],
) -> qt.Qobj:
    dim = subsystem.truncated_dim

    if op_in_eigenbasis is False:
        if evecs is None:
            _, evecs = subsystem.eigensys(evals_count=dim)
        operator_matrixelements = get_matrixelement_table(operator, evecs)
        return qt.Qobj(operator_matrixelements)
    return qt.Qobj(operator[:dim, :dim])


def convert_opstring_to_qobj(
    operator: str,
    subsystem: Union["QubitBaseClass", "Oscillator"],
    evecs: Optional[bc.backend.ndarray],
) -> qt.Qobj:
    dim = subsystem.truncated_dim

    if evecs is None:
        _, evecs = subsystem.eigensys(evals_count=dim)
    operator_matrixelements = subsystem.matrixelement_table(operator, evecs=evecs)
    return qt.Qobj(operator_matrixelements)


def convert_operator_to_qobj(
    operator: Union[bc.backend.ndarray, bc.backend.csc_matrix, bc.backend.dia_matrix, qt.Qobj, str],
    subsystem: Union["QubitBaseClass", "Oscillator"],
    op_in_eigenbasis: bool,
    evecs: Optional[bc.backend.ndarray],
) -> qt.Qobj:
    if isinstance(operator, qt.Qobj):
        return operator
    if isinstance(operator, (bc.backend.ndarray, bc.backend.csc_matrix, bc.backend.csr_matrix, bc.backend.dia_matrix)):
        return convert_matrix_to_qobj(operator, subsystem, op_in_eigenbasis, evecs)
    if isinstance(operator, str):
        return convert_opstring_to_qobj(operator, subsystem, evecs)
    raise TypeError("Unsupported operator type: ", type(operator))


def generate_target_states_list(
    sweep: "ParameterSweep", initial_state_labels: Tuple[bc.backend.int_, ...]
) -> List[Tuple[bc.backend.int_, ...]]:
    """Based on a bare state label (i1, i2, ...)  with i1 being the excitation level
    of subsystem 1, i2 the excitation level of subsystem 2 etc., generate a list of
    new bare state labels. These bare state labels correspond to target states
    reached from the given initial one by single-photon qubit transitions. These are
    transitions where one of the qubit excitation levels increases at a time. There
    are no changes in oscillator photon numbers.

    Parameters
    ----------
    sweep:
    initial_state_labels:
        bare-state labels of the initial state whose energy is supposed to be subtracted
        from the spectral data
    """
    target_states_list = []
    for qbt_subsys in sweep.qbt_subsys_list:  # iterate through qubit subsys_list
        assert qbt_subsys.truncated_dim is not None
        subsys_index = sweep._hilbertspace.get_subsys_index(qbt_subsys)
        initial_qbt_state = initial_state_labels[subsys_index]
        for state_label in range(initial_qbt_state + 1, qbt_subsys.truncated_dim):
            # for given qubit subsystem, generate target labels by increasing that qubit
            # excitation level
            target_labels = list(initial_state_labels)
            target_labels[subsys_index] = state_label
            target_states_list.append(tuple(target_labels))
    return target_states_list


def recast_esys_mapdata(
    esys_mapdata: List[Tuple[bc.backend.ndarray, bc.backend.ndarray]]
) -> Tuple[bc.backend.ndarray, List[bc.backend.ndarray]]:
    """
    Takes data generated by a map of eigensystem calls and returns the eigenvalue and
    eigenstate tables

    Returns
    -------
        eigenvalues and eigenvectors
    """
    paramvals_count = len(esys_mapdata)
    eigenenergy_table = bc.backend.asarray(
        [esys_mapdata[index][0] for index in range(paramvals_count)]
    )
    eigenstate_table = [esys_mapdata[index][1] for index in range(paramvals_count)]
    return eigenenergy_table, eigenstate_table


def identity_wrap(
    operator: Union[str, bc.backend.ndarray, Qobj, Callable],
    subsystem: "QuantumSys",
    subsys_list: List["QuantumSys"],
    op_in_eigenbasis: bool = False,
    evecs: Optional[bc.backend.ndarray] = None,
) -> Qobj:
    """Takes the `operator` belonging to `subsystem` and "wraps" it in identities.
    The full Hilbert space is taken to consist of all subsystems given as
    `subsys_list`. `subsystem` must be one element in that list. For each of the
    other subsystems in the list, an identity operator of the correct dimension is
    generated and inserted into the appropriate Kronecker product "sandwiching" the
    operator.

    Parameters
    ----------
    operator:
        operator acting in Hilbert space of `subsystem`; if str, then this should be an
        operator name in the subsystem, typically not in eigenbasis
    subsystem:
        subsystem where diagonal operator is defined
    subsys_list:
        list of all subsystems relevant to the Hilbert space.
    op_in_eigenbasis:
        whether `operator` is given in the `subsystem` eigenbasis; otherwise,
        `operator` is assumed to be in the internal QuantumSystem basis. This
        argument is ignored if `operator` is given as a Qobj.
    evecs:
        internal `QuantumSystem` eigenstates, used to convert `operator` into eigenbasis

    Returns
    -------
        operator in the full Hilbert space (as specified by `subsystem_list`). This
        operator is expressed in the bare product basis consisting of the energy
        eigenstates of each subsystem (unless `operator` is provided as a `Qobj`,
        in which case no conversion takes place).
    """
    if not isinstance(operator, qt.Qobj) and callable(operator):
        try:
            operator = operator(energy_esys=(None, evecs))
        except TypeError:
            operator = operator()
        op_in_eigenbasis = True

    subsys_operator = convert_operator_to_qobj(
        operator, subsystem, op_in_eigenbasis, evecs  # type:ignore
    )
    operator_identitywrap_list = [
        qt.operators.qeye(the_subsys.truncated_dim) for the_subsys in subsys_list
    ]
    subsystem_index = subsys_list.index(subsystem)
    operator_identitywrap_list[subsystem_index] = subsys_operator
    return qt.tensor(operator_identitywrap_list)
