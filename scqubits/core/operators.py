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
from scqubits import backend_change

def annihilation(dimension: int) -> ndarray:
    """
    Returns a dense matrix of size dimension x dimension representing the annihilation
    operator in number basis.
    """
    middle_variable = backend_change.backend.array(range(1, dimension))
    offdiag_elements = backend_change.backend.sqrt(middle_variable)
    return backend_change.backend.diagflat(offdiag_elements, 1)


def annihilation_sparse(dimension: int) -> csc_matrix:
    """Returns a matrix of size dimension x dimension representing the annihilation
    operator in the format of a scipy sparse.csc_matrix.
    """

    offdiag_elements = backend_change.backend.sqrt(backend_change.backend.arange(dimension))
        
    return sp.sparse.dia_matrix(
        (offdiag_elements, [1]), shape=(dimension, dimension)
    ).tocsc()


def creation(dimension: int) -> ndarray:
    """
    Returns a dense matrix of size dimension x dimension representing the creation
    operator in number basis.
    """
    return annihilation(dimension).T


def creation_sparse(dimension: int) -> csc_matrix:
    """Returns a matrix of size dimension x dimension representing the creation operator
    in the format of a scipy sparse.csc_matrix
    """
    return annihilation_sparse(dimension).transpose().tocsc()


def hubbard_sparse(j1: int, j2: int, dimension: int) -> csc_matrix:
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
    hubbardmat = sp.sparse.dok_matrix((dimension, dimension), dtype=backend_change.backend.float_)
    hubbardmat[j1, j2] = 1.0
    return hubbardmat.asformat("csc")


def number(
    dimension: int, prefactor: Optional[Union[float, complex]] = None
) -> ndarray:
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
    diag_elements = backend_change.backend.arange(dimension, dtype=backend_change.backend.float_)
    if prefactor:
        diag_elements *= prefactor
    return backend_change.backend.diagflat(diag_elements)


def number_sparse(
    dimension: int, prefactor: Optional[Union[float, complex]] = None
) -> csc_matrix:
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
    diag_elements = backend_change.backend.arange(dimension, dtype=backend_change.backend.float_)
    if prefactor:
        diag_elements *= prefactor
    return sp.sparse.dia_matrix(
        (diag_elements, [0]), shape=(dimension, dimension), dtype=backend_change.backend.float_
    ).tocsc()


def a_plus_adag_sparse(
    dimension: int, prefactor: Union[float, complex, None] = None
) -> csc_matrix:
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
    dimension: int, prefactor: Union[float, complex, None] = None
) -> ndarray:
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
    return a_plus_adag_sparse(dimension, prefactor=prefactor).toarray()


def cos_theta_harmonic(
    dimension: int, prefactor: Union[float, complex, None] = None
) -> ndarray:
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
    return sp.linalg.cosm(a_plus_adag_sparse(dimension, prefactor=prefactor).toarray())


def sin_theta_harmonic(
    dimension: int, prefactor: Union[float, complex, None] = None
) -> ndarray:
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
    return sp.linalg.sinm(a_plus_adag_sparse(dimension, prefactor=prefactor).toarray())


def iadag_minus_ia_sparse(
    dimension: int, prefactor: Union[float, complex, None] = None
) -> csc_matrix:
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
    dimension: int, prefactor: Union[float, complex, None] = None
) -> ndarray:
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
    return iadag_minus_ia_sparse(dimension, prefactor=prefactor).toarray()


def sigma_minus() -> backend_change.backend.ndarray:
    return sigma_plus().T


def sigma_plus() -> backend_change.backend.ndarray:
    return backend_change.backend.asarray([[0.0, 1.0], [0.0, 0.0]])


def sigma_x() -> backend_change.backend.ndarray:
    return backend_change.backend.asarray([[0.0, 1.0], [1.0, 0.0]])


def sigma_y() -> backend_change.backend.ndarray:
    return backend_change.backend.asarray([[0.0, -1j], [1j, 0.0]])


def sigma_z() -> backend_change.backend.ndarray:
    return backend_change.backend.asarray([[1.0, 0.0], [0.0, -1.0]])