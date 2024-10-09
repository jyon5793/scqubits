# operators.py
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

from typing import Optional, Union

import numpy as np
import scipy as sp

from numpy import ndarray
from scipy.sparse import csc_matrix
from scqubits import backend_change as bc

def annihilation(dimension: bc.backend.int_) -> bc.backend.ndarray:
    """
    Returns a dense matrix of size dimension x dimension representing the annihilation
    operator in number basis.
    """
    middle_variable = bc.backend.array(range(1, dimension))
    offdiag_elements = bc.backend.sqrt(middle_variable)
    return bc.backend.diagflat(offdiag_elements, 1)


def annihilation_sparse(dimension: bc.backend.int) ->bc.backend.csc_matrix:
    """Returns a matrix of size dimension x dimension representing the annihilation
    operator in the format of a scipy sparse.csc_matrix.
    """

    offdiag_elements = bc.backend.sqrt(bc.backend.arange(dimension))
        
    return bc.backend.solve_csc_matrix(bc.backend.dia_matrix(
        (offdiag_elements, [1]), shape=(dimension, dimension)
    ))


def creation(dimension: bc.backend.int_) -> bc.backend.ndarray:
    """
    Returns a dense matrix of size dimension x dimension representing the creation
    operator in number basis.
    """
    return annihilation(dimension).T


def creation_sparse(dimension: bc.backend.int_) -> bc.backend.csc_matrix:
    """Returns a matrix of size dimension x dimension representing the creation operator
    in the format of a scipy sparse.csc_matrix
    """
    return bc.backend.solve_csc_matrix(annihilation_sparse(dimension).transpose())


def hubbard_sparse(j1: bc.backend.int_, j2: bc.backend.int_, dimension: bc.backend.int_) -> bc.backend.csc_matrix:
    """The Hubbard operator :math:`|j1\\rangle>\\langle j2|` is returned as a matrix of
    linear size dimension.

    Parameters
    ----------
    dimension:
    j1, j2:
        indices of the two states labeling the Hubbard operator

    Returns
    -------
        sparse number operator matrix, size dimension x dimension
    """
    hubbardmat = bc.backend.dok_matrix((dimension, dimension), dtype=bc.backend.float_)
    hubbardmat[j1, j2] = 1.0
    return bc.backend.asformat(hubbardmat, "csc")


def number(
    dimension: bc.backend.int_, prefactor: Optional[Union[bc.backend.float_, bc.backend.complex_]] = None
) -> bc.backend.ndarray:
    """Number operator matrix of size dimension x dimension in sparse matrix
    representation. An additional prefactor can be directly included in the
    generation of the matrix by supplying 'prefactor'.

    Parameters
    ----------
    dimension:
        matrix dimension
    prefactor:
        prefactor multiplying the number operator matrix


    Returns
    -------
        number operator matrix, size dimension x dimension
    """
    diag_elements = bc.backend.arange(dimension, dtype=bc.backend.float_)
    if prefactor:
        diag_elements *= prefactor
    return bc.backend.diagflat(diag_elements)


def number_sparse(
    dimension: bc.backend.int_, prefactor: Optional[Union[bc.backend.float_, bc.backend.complex_]] = None
) -> bc.backend.csc_matrix:
    """Number operator matrix of size dimension x dimension in sparse matrix
    representation. An additional prefactor can be directly included in the
    generation of the matrix by supplying 'prefactor'.

    Parameters
    ----------
    dimension:
        matrix size
    prefactor:
        prefactor multiplying the number operator matrix

    Returns
    -------
        sparse number operator matrix, size dimension x dimension
    """
    diag_elements = bc.backend.arange(dimension, dtype=bc.backend.float_)
    if prefactor:
        diag_elements *= prefactor
    return bc.backend.solve_csc_matrix(bc.backend.dia_matrix(
        (diag_elements, [0]), shape=(dimension, dimension), dtype=bc.backend.float_
    ))


def a_plus_adag_sparse(
    dimension: bc.backend.int_, prefactor: Union[bc.backend.float_, bc.backend.complex_, None] = None
) -> bc.backend.csc_matrix:
    """Operator matrix for prefactor(a+a^dag) of size dimension x dimension in
    sparse matrix representation.

    Parameters
    ----------
    dimension:
        matrix size
    prefactor:
        prefactor multiplying the number operator matrix
        (if not given, this defaults to 1)

    Returns
    -------
        prefactor * (a + a^dag) as sparse operator matrix, size dimension x dimension
    """
    prefactor = prefactor if prefactor is not None else 1.0
    return prefactor * (annihilation_sparse(dimension) + creation_sparse(dimension))


def a_plus_adag(
    dimension: bc.backend.int_, prefactor: Union[bc.backend.float_, bc.backend.complex_, None] = None
) -> bc.backend.ndarray:
    """Operator matrix for prefactor(a+a^dag) of size dimension x dimension in
    sparse matrix representation.

    Parameters
    ----------
    dimension:
        matrix size
    prefactor:
        prefactor multiplying the number operator matrix
        (if not given, this defaults to 1)

    Returns
    -------
        prefactor (a + a^dag) as ndarray, size dimension x dimension
    """
    return bc.backend.toarray(a_plus_adag_sparse(dimension, prefactor=prefactor))


def cos_theta_harmonic(
    dimension: bc.backend.int_, prefactor: Union[bc.backend.float_, bc.backend.complex_, None] = None
) -> bc.backend.ndarray:
    """Operator matrix for cos(prefactor(a+a^dag)) of size dimension x dimension in
    sparse matrix representation.

    Parameters
    ----------
    dimension:
        matrix size
    prefactor:
        prefactor multiplying the number operator matrix
        (if not given, this defaults to 1)

    Returns
    -------
        prefactor (a + a^dag) as ndarray, size dimension x dimension
    """
    return bc.backend.spcosm(bc.backend.toarray(a_plus_adag_sparse(dimension, prefactor=prefactor)))


def sin_theta_harmonic(
    dimension: bc.backend.int_, prefactor: Union[bc.backend.float_, bc.backend.complex_, None] = None
) -> bc.backend.ndarray:
    """Operator matrix for sin(prefactor(a+a^dag)) of size dimension x dimension in
    sparse matrix representation.

    Parameters
    ----------
    dimension:
        matrix size
    prefactor:
        prefactor multiplying the number operator matrix
        (if not given, this defaults to 1)

    Returns
    -------
        prefactor (a + a^dag) as ndarray, size dimension x dimension
    """
    return bc.backend.spsinm(a_plus_adag_sparse(dimension, prefactor=prefactor).toarray())


def iadag_minus_ia_sparse(
    dimension: bc.backend.int_, prefactor: Union[bc.backend.float_, bc.backend.complex_, None] = None
) -> bc.backend.csc_matrix:
    """Operator matrix for prefactor(ia-ia^dag) of size dimension x dimension as
    ndarray

    Parameters
    ----------
    dimension:
        matrix size
    prefactor:
        prefactor multiplying the number operator matrix
        (if not given, this defaults to 1)

    Returns
    -------
        prefactor  (ia - ia^dag) as sparse operator matrix, size dimension x dimension
    """
    prefactor = prefactor if prefactor is not None else 1.0
    return prefactor * (
        1j * creation_sparse(dimension) - 1j * annihilation_sparse(dimension)
    )


def iadag_minus_ia(
    dimension: bc.backend.int_, prefactor: Union[bc.backend.float_, bc.backend.complex_, None] = None
) -> bc.backend.ndarray:
    """Operator matrix for prefactor(ia-ia^dag) of size dimension x dimension as
    ndarray

    Parameters
    ----------
    dimension:
        matrix size
    prefactor:
        prefactor multiplying the number operator matrix
        (if not given, this defaults to 1)

    Returns
    -------
        prefactor  (ia - ia^dag) as ndarray, size dimension x dimension
    """
    return bc.backend.toarray(iadag_minus_ia_sparse(dimension, prefactor=prefactor))


def sigma_minus() -> bc.backend.ndarray:
    return sigma_plus().T


def sigma_plus() -> bc.backend.ndarray:
    return bc.backend.asarray([[0.0, 1.0], [0.0, 0.0]])


def sigma_x() -> bc.backend.ndarray:
    return bc.backend.asarray([[0.0, 1.0], [1.0, 0.0]])


def sigma_y() -> bc.backend.ndarray:
    return bc.backend.asarray([[0.0, -1j], [1j, 0.0]])


def sigma_z() -> bc.backend.ndarray:
    return bc.backend.asarray([[1.0, 0.0], [0.0, -1.0]])