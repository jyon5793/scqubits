import numpy as np
import jax
import scipy
from functools import wraps
from scipy import sparse
from jax import custom_vjp
from scipy.special import pbdv
import scipy as sp
import jax.experimental.sparse as jsp
from jax import jit
import sympy
import sympy2jax

class Backend(object):
    int = np.int64
    float = np.float64
    complex = np.complex128
    ndarray = np.ndarray

    def __repr__(self):
        return self.__class__.__name__
    
    @staticmethod
    def convert_to_array(obj_list):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @staticmethod
    def eye(N, dtype=None):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    @staticmethod
    def fill_diagonal(array, value):
        raise NotImplementedError("This method should be overridden by subclasses")
    
class NumpyBackend(Backend):
    # numpy methods
    __name__ = 'numpy'
    sympy = sympy
    dtype = staticmethod(np.dtype)
    array = staticmethod(np.array)
    dot = staticmethod(np.dot)
    vdot = staticmethod(np.vdot)
    exp = staticmethod(np.exp)
    sqrt = staticmethod(np.sqrt)
    pi = staticmethod(np.pi)
    sin = staticmethod(np.sin)
    cos = staticmethod(np.cos)
    square = staticmethod(np.square)
    arange = staticmethod(np.arange)
    linspace = staticmethod(np.linspace)
    outer = staticmethod(np.outer)
    matmul = staticmethod(np.matmul)
    sort = staticmethod(np.sort)
    ndarray = staticmethod(np.ndarray)
    diagflat = staticmethod(np.diagflat)
    zeros = staticmethod(np.zeros)
    prod = staticmethod(np.prod)
    empty = staticmethod(np.empty)
    asarray = staticmethod(np.asarray)
    float64 = staticmethod(np.float64)
    sum = staticmethod(np.sum)
    linalg = staticmethod(np.linalg)
    ones = staticmethod(np.ones)
    diag = staticmethod(np.diag)
    zeros_like = staticmethod(np.zeros_like)
    isrealobj = staticmethod(np.isrealobj)
    fromiter = staticmethod(np.fromiter)
    identity = staticmethod(np.identity)
    argsort = staticmethod(np.argsort)
    meshgrid = staticmethod(np.meshgrid)
    einsum = staticmethod(np.einsum)
    real = staticmethod(np.real)
    abs = staticmethod(np.abs)
    imag = staticmethod(np.imag)
    diagonal = staticmethod(np.diagonal)
    float_ = staticmethod(np.float_)
    inf = staticmethod(np.inf)
    sinh = staticmethod(np.sinh)
    tanh = staticmethod(np.tanh)
    tensordot = staticmethod(np.tensordot)
    transpose = staticmethod(np.transpose)
    complex_ = staticmethod(np.complex_)
    int_ = staticmethod(np.int_)
    float32 = staticmethod(np.float32)
    integer = staticmethod(np.integer)
    index_exp = staticmethod(np.index_exp)
    all = staticmethod(np.all)
    log = staticmethod(np.log)
    fill_diagonal = staticmethod(np.fill_diagonal)
    diag_indices = staticmethod(np.diag_indices)
    isnan = staticmethod(np.isnan)
    empty_like = staticmethod(np.empty_like)
    repeat = staticmethod(np.repeat)
    unique = staticmethod(np.unique)
    round = staticmethod(np.round)
    where = staticmethod(np.where)
    isinf = staticmethod(np.isinf)
    intersect1d = staticmethod(np.intersect1d)
    vstack = staticmethod(np.vstack)
    kron = staticmethod(np.kron)
    reshape = staticmethod(np.reshape)
    complex64 = staticmethod(np.complex64)
    nan = staticmethod(np.nan)
    max = staticmethod(np.max)
    min = staticmethod(np.min)
    full = staticmethod(np.full)
    pad = staticmethod(np.pad)
    isclose = staticmethod(np.isclose)
    argmax = staticmethod(np.argmax)
    unravel_index = staticmethod(np.unravel_index)
    sign = staticmethod(np.sign)
    allclose = staticmethod(np.allclose)
    ndenumerate = staticmethod(np.ndenumerate)
    stack = staticmethod(np.stack)
    eigvalsh = staticmethod(scipy.linalg.eigvalsh)
    eigh = staticmethod(scipy.linalg.eigh)
    eig = staticmethod(sp.linalg.eig)
    csc_matrix = staticmethod(scipy.sparse.csc_matrix)
    dia_matrix = staticmethod(scipy.sparse.dia_matrix)
    coo_matrix = staticmethod(scipy.sparse.coo_matrix)
    dok_matrix = staticmethod(scipy.sparse.dok_matrix)
    csr_matrix = staticmethod(scipy.sparse.csr_matrix)
    spidentity = staticmethod(scipy.sparse.identity)
    spkron = staticmethod(scipy.sparse.kron)
    expm = staticmethod(scipy.linalg.expm)
    spexpm = staticmethod(scipy.sparse.linalg.expm)
    diags = staticmethod(scipy.sparse.diags)
    block_diag = staticmethod(scipy.linalg.block_diag)
    special_kv = staticmethod(scipy.special.kv)
    spcosm = staticmethod(scipy.linalg.cosm)
    spsinm = staticmethod(scipy.linalg.sinm)
    speye = staticmethod(scipy.sparse.eye)
    speigsh = staticmethod(scipy.sparse.linalg.eigsh)
    qr = staticmethod(scipy.linalg.qr)
    inv = staticmethod(np.linalg.inv)
    constants_hbar = staticmethod(scipy.constants.hbar)
    constants_k = staticmethod(scipy.constants.k)
    constants_e = staticmethod(scipy.constants.e)
    constants_h = staticmethod(scipy.constants.h)
    matrix_rank = staticmethod(np.linalg.matrix_rank)
    det = staticmethod(np.linalg.det)
    pinv = staticmethod(np.linalg.pinv)
    delete = staticmethod(np.delete)
    eigvalsh_tridiagonal = staticmethod(scipy.linalg.eigvalsh_tridiagonal)
    eigh_tridiagonal = staticmethod(sp.linalg.eigh_tridiagonal)

    @staticmethod
    def toarray(matrix):
        return matrix.toarray()

    @staticmethod
    def asformat(matrix, format):
        return matrix.asformat(format)
    
    @staticmethod
    def to_csc_matrix(matrix):
        # 如果输入是稠密矩阵或其他类型，转换为 CSC 格式
        if isinstance(matrix, np.ndarray):
            return sparse.csc_matrix(matrix)
        # 如果输入是稀疏矩阵，调用 .tocsc() 方法转换为 CSC 格式
        elif sparse.issparse(matrix):
            return matrix.tocsc()
        # 处理其他情况，如果输入既不是稠密矩阵，也不是稀疏矩阵
        else:
            raise TypeError(f"Unsupported matrix type: {type(matrix)}. Expected a dense matrix or a sparse matrix.")

    @staticmethod
    def convert_to_array(obj_list):
        return np.array(obj_list, dtype=object)
    @staticmethod
    def eye(N, dtype=None):
        return np.eye(N, dtype=dtype)
    
    @staticmethod
    def array_solve(arr, index_tuple, value):
        arr[index_tuple] = value
        return arr
    
    @staticmethod
    def bind_custom_vjp(fwd_func, bwd_func, func):
        return func
    
    @staticmethod
    def solve_pure_callback(method, matrix_shape, dtype=np.float_):
        return method()

    @staticmethod
    def is_sparse(matrix):
        if sp.sparse.issparse(matrix):
            return True
        else:
            return False
        
    @staticmethod
    def to_dense(matrix):
        if sp.sparse.issparse(matrix): 
            return matrix.toarray()
        return matrix
        
    @staticmethod
    def to_sparse(matrix):
        return scipy.sparse.csc_matrix(matrix)


class JaxBackend(Backend):
    __name__ = 'jax'
    scipy = sympy2jax
    dtype = staticmethod(jax.numpy.dtype)
    ndarray = staticmethod(jax.numpy.array)
    int = staticmethod(jax.numpy.int64)
    array = staticmethod(jax.numpy.array)
    dot = staticmethod(jax.numpy.dot)
    vdot = staticmethod(jax.numpy.vdot)
    exp = staticmethod(jax.numpy.exp)
    sqrt = staticmethod(jax.numpy.sqrt)
    pi = staticmethod(jax.numpy.pi)
    sin = staticmethod(jax.numpy.sin)
    cos = staticmethod(jax.numpy.cos)
    square = staticmethod(jax.numpy.square)
    arange = staticmethod(jax.numpy.arange)
    linspace = staticmethod(jax.numpy.linspace)
    outer = staticmethod(jax.numpy.outer)
    matmul = staticmethod(jax.numpy.matmul)
    sort = staticmethod(jax.numpy.sort)
    diagflat = staticmethod(jax.numpy.diagflat)
    zeros = staticmethod(jax.numpy.zeros)
    prod = staticmethod(jax.numpy.prod)
    empty = staticmethod(jax.numpy.empty)
    asarray = staticmethod(jax.numpy.asarray)
    float64 = staticmethod(jax.numpy.float64)
    sum = staticmethod(jax.numpy.sum)
    linalg = staticmethod(jax.numpy.linalg)
    ones = staticmethod(jax.numpy.ones)
    diag = staticmethod(jax.numpy.diag)
    zeros_like = staticmethod(jax.numpy.zeros_like)
    isrealobj = staticmethod(jax.numpy.isrealobj)
    fromiter = staticmethod(jax.numpy.fromiter)
    identity = staticmethod(jax.numpy.identity)
    argsort = staticmethod(jax.numpy.argsort)
    meshgrid = staticmethod(jax.numpy.meshgrid)
    einsum = staticmethod(jax.numpy.einsum)
    real = staticmethod(jax.numpy.real)
    imag = staticmethod(jax.numpy.imag)
    diagonal = staticmethod(jax.numpy.diagonal)
    float_ = staticmethod(jax.numpy.float_)
    inf = staticmethod(jax.numpy.inf)
    sinh = staticmethod(jax.numpy.sinh)
    tanh = staticmethod(jax.numpy.tanh)
    tensordot = staticmethod(jax.numpy.tensordot)
    transpose = staticmethod(jax.numpy.transpose)
    complex_ = staticmethod(jax.numpy.complex_)
    float32 = staticmethod(jax.numpy.float32)
    int_ = staticmethod(jax.numpy.int_)
    integer = staticmethod(jax.numpy.integer)
    index_exp = staticmethod(jax.numpy.index_exp)
    all = staticmethod(jax.numpy.all)
    log = staticmethod(jax.numpy.log)
    # fill_diagonal = staticmethod(lambda array, value: array.at[jax.numpy.diag_indices(min(array.shape))].set(value))
    diag_indices = staticmethod(jax.numpy.diag_indices)
    isnan = staticmethod(jax.numpy.isnan)
    empty_like = staticmethod(jax.numpy.empty_like)
    repeat = staticmethod(jax.numpy.repeat)
    unique = staticmethod(jax.numpy.unique)
    round = staticmethod(jax.numpy.round)
    where = staticmethod(jax.numpy.where)
    isinf = staticmethod(jax.numpy.isinf)
    intersect1d = staticmethod(jax.numpy.intersect1d)
    vstack = staticmethod(jax.numpy.vstack)
    kron = staticmethod(jax.numpy.kron)
    reshape = staticmethod(jax.numpy.reshape)
    complex64 = staticmethod(jax.numpy.complex64)
    nan = staticmethod(jax.numpy.nan)
    max = staticmethod(jax.numpy.max)
    min = staticmethod(jax.numpy.min)
    abs = staticmethod(jax.numpy.abs)
    full = staticmethod(jax.numpy.full)
    pad= staticmethod(jax.numpy.pad)
    isclose = staticmethod(jax.numpy.isclose)
    argmax = staticmethod(jax.numpy.argmax)
    unravel_index = staticmethod(jax.numpy.unravel_index)
    sign = staticmethod(jax.numpy.sign)
    allclose = staticmethod(jax.numpy.allclose)
    stack = staticmethod(jax.numpy.stack)
    custom_vjp = staticmethod(jax.custom_jvp)
    csc_matrix = staticmethod(jsp.CSC)
    coo_matrix = staticmethod(jsp.COO)
    csr_matrix = staticmethod(jsp.CSR)
    scipy = staticmethod(jax.scipy)
    grad = staticmethod(jax.grad)
    value_and_grad = staticmethod(jax.value_and_grad)
    eigh = staticmethod(jax.scipy.linalg.eigh)
    eig = staticmethod(jax.numpy.linalg.eig)
    # eigvalsh = staticmethod(jax.scipy.linalg.eigh)
    expm = staticmethod(jax.scipy.linalg.expm)
    block_diag = staticmethod(jax.scipy.linalg.block_diag)
    qr = staticmethod(jax.scipy.linalg.qr)
    inv = staticmethod(jax.numpy.linalg.inv)
    constants_hbar = 1.054571817e-34
    constants_k = 1.380649e-23
    constants_e = 1.602176634e-19
    constants_h = 6.62607015e-34
    matrix_rank = staticmethod(jax.numpy.linalg.matrix_rank)
    det = staticmethod(jax.numpy.linalg.det)
    pinv = staticmethod(jax.numpy.linalg.pinv)
    delete = staticmethod(jax.numpy.delete)
    eigh_tridiagonal = staticmethod(jax.scipy.linalg.eigh_tridiagonal)
    
    @staticmethod
    def eigvalsh_tridiagonal(A):
        return jax.scipy.linalg.eigh_tridiagonal(A)[0]
    
    @staticmethod
    def eigvalsh(A):
        return jax.scipy.linalg.eigh(A)[0]  # 只返回特征值部分
    
    @staticmethod
    def spsinm(A):
        iA = 1j * A
        exp_iA = jax.scipy.linalg.expm(iA)
        exp_neg_iA = jax.scipy.linalg.expm(-iA)
        return (exp_iA - exp_neg_iA) / (2.0 * 1j)

    @staticmethod
    def spcosm(A):
        iA = 1j * A
        exp_iA = jax.scipy.linalg.expm(iA)
        exp_neg_iA = jax.scipy.linalg.expm(-iA)
        return (exp_iA + exp_neg_iA) / 2.0

    @staticmethod
    def toarray(matrix):
        if isinstance(matrix, jsp.BCOO):
            return matrix.todense()
        return matrix
    
    @staticmethod
    def spidentity(n,format):
         # 构造单位矩阵的非零元素（对角线上的元素全为1）
        data = jax.numpy.ones(n)
        indices = jax.numpy.arange(n)[:, None]  # 行和列的索引是相同的
        indices = jax.numpy.hstack([indices, indices])  # 构造 (row, col) 索引对
        if format == "csc":
            # 如果 format 是 "csc"，创建 CSC 稀疏矩阵
            return jsp.CSC((data, indices), shape=(n, n))
        elif format == "array":
            return jsp.BCOO((data, indices), shape=(n, n)).todense

        # 创建 BCOO 稀疏单位矩阵
        return jsp.BCOO((data, indices), shape=(n, n))

    @staticmethod
    def spkron(a: jsp.BCOO, b: jsp.BCOO, format=None):
        # 获取 a 和 b 的数据及索引
        a_data, a_indices = a.data, a.indices
        b_data, b_indices = b.data, b.indices

        # 获取形状
        a_shape = a.shape
        b_shape = b.shape

        # Kronecker 积的形状
        result_shape = (a_shape[0] * b_shape[0], a_shape[1] * b_shape[1])

        # 构建 Kronecker 积的数据和索引
        result_data = jax.numpy.kron(a_data, b_data)

        row_indices_a = a_indices[:, 0][:, None]
        col_indices_a = a_indices[:, 1][:, None]
        row_indices_b = b_indices[:, 0]
        col_indices_b = b_indices[:, 1]

        result_indices_row = row_indices_a * b_shape[0] + row_indices_b
        result_indices_col = col_indices_a * b_shape[1] + col_indices_b
        result_indices = jax.numpy.hstack([result_indices_row, result_indices_col])

        # 根据 `format` 参数，返回不同的矩阵类型
        if format == "csc":
            return jsp.CSC((result_data, result_indices), shape=result_shape)
        elif format == "array":
            return jsp.BCOO((result_data, result_indices), shape=result_shape).todense()
        else:
            return jsp.BCOO((result_data, result_indices), shape=result_shape)

    @staticmethod
    def convert_to_array(obj_list):
        return obj_list  # 对于JAX，不转换为数组
    
    @staticmethod
    def eye(N, dtype=None):
        return jax.numpy.eye(N, dtype=dtype)
    
    @staticmethod
    def fill_diagonal(array, value):
        indices = jax.numpy.diag_indices(min(array.shape))
        return array.at[indices].set(value)
    
    @staticmethod
    def array_solve(arr, index_tuple, value):
        arr = arr.at[index_tuple].set(value)
        return arr
    
    @staticmethod
    def bind_custom_vjp(fwd_func, bwd_func, func):
        custom_vjp_func = custom_vjp(func)
        custom_vjp_func.defvjp(fwd_func, bwd_func)
        return custom_vjp_func
    
    @staticmethod
    def is_sparse(matrix):
        if isinstance(matrix, jsp.BCOO) or isinstance(matrixt, jsp.COO) or isinstance(matrix, jsp.CSC):
            return True
        else:
            return False

    @staticmethod
    def to_dense(matrix):
        if isinstance(matrix, jsp.BCOO) or isinstance(matrixt, jsp.COO) or isinstance(matrix, jsp.CSC):
            return matrix.todense()
        return matrix
        
    @staticmethod
    def to_sparse(matrix):
        return jsp.BCOO.fromdense(matrix)


    @staticmethod
    def solve_pure_callback(method, matrix_shape, dtype=jax.numpy.float_):
        """
        通用的纯回调函数，用于处理 JAX 的 pure_callback 或直接返回 NumPy 后端结果。

        Parameters
        ----------
        method: callable
            一个方法或函数，执行需要包装的逻辑（例如 SciPy 转换操作）。
        matrix_shape: tuple
            矩阵的形状。
        dtype: jnp.dtype, optional
            返回矩阵的类型，默认是 jnp.float_。

        Returns
        -------
        result: jax.BCOO 或 ndarray
            处理后的矩阵，稀疏矩阵或稠密矩阵，取决于后端。
        """
        return jax.pure_callback(
                method, 
                jax.ShapeDtypeStruct(matrix_shape, dtype)
            )
    
    @staticmethod
    def dia_matrix(diagonal, shape, dtype=None): 
        """
        在 JAX 中构建对角稀疏矩阵，类似于 SciPy 的 sparse.dia_matrix。

        Parameters
        ----------
        diagonal : array
            对角线上非零元素的值。
        shape : tuple
            稀疏矩阵的形状。
        dtype : data-type, optional
            数据类型。如果未指定，将使用默认的 JAX 数据类型。

        Returns
        -------
        BCOO 稀疏矩阵
        """
        # 非零元素为对角线上的元素，应用 dtype
        data = jax.numpy.array(diagonal, dtype=dtype)

        # 对角线元素的位置 (row, col)
        indices = jax.numpy.arange(len(diagonal), dtype=jax.numpy.int32)
        indices = jax.numpy.stack([indices, indices], axis=1)  # 构造 (row, col) 索引对

        # 构建稀疏 BCOO 矩阵
        return jsp.BCOO((data, indices), shape=shape)
    
    @staticmethod
    def dok_matrix(shape, dtype=None):
        """Create a sparse BCOO matrix with zeros in JAX, similar to a DOK matrix."""
        data = jax.numpy.array([], dtype=dtype or jax.numpy.float64)
        indices = jax.numpy.empty((0, 2), dtype=jax.numpy.int32)
        return jsp.BCOO((data, indices), shape=shape)
    
    @staticmethod
    def to_csc_matrix(matrix):
        if isinstance(matrix, scipy.sparse.csc_matrix):
            # Extract data, indices, and indptr from the CSC matrix
            data = matrix.data
            indices = matrix.indices
            indptr = matrix.indptr
            shape = matrix.shape

            # Convert CSC to COO
            rows = []
            for col in range(len(indptr) - 1):
                start_idx = indptr[col]
                end_idx = indptr[col + 1]
                rows.extend([col] * (end_idx - start_idx))
            
            # Stack row and column indices
            rows = jax.numpy.array(rows)
            cols = jax.numpy.array(indices)
            coo_indices = jax.numpy.stack([rows, cols], axis=1)

            # Create BCOO sparse matrix in JAX
            data = jax.numpy.array(data)
            bcoo_matrix = jsp.BCOO((data, coo_indices), shape=shape)
            # If format is "csc", create a CSC sparse matrix
            return jsp.CSC((data, coo_indices), shape=shape)
            
        elif isinstance(matrix, jsp.CSC) or isinstance(matrix, jsp.BCOO):
            # If it's already a jsp.CSC matrix, return it as-is
            return jsp.CSC(matrix.data, shape=matrix.shape)
        else:
            # If input is not a recognized sparse type, raise an error
            raise TypeError("Input matrix must be either a scipy.sparse.csc_matrix or a JAX BCOO matrix.")

    @staticmethod
    def spexpm(A):
        v = jax.numpy.ones(A.shape[0])  # 这里使用一个全1的单位向量作为 Krylov 子空间的初始向量
        m = 10 
        return krylov_expm_jax(A,v,m)
    
    @staticmethod
    def diags(diagonals, offsets, shape):
        data = []
        indices = []

        for diag, offset in zip(diagonals, offsets):
            n = len(diag)
            if offset >= 0:
                row = jax.numpy.arange(n)
                col = row + offset
            else:
                col = jax.numpy.arange(n)
                row = col - offset
            
            # 过滤超出矩阵大小的元素
            valid = (row < shape[0]) & (col < shape[1])
            data.append(jax.numpy.array(diag)[valid])
            indices.append(jax.numpy.stack([row[valid], col[valid]], axis=1))

        data = jax.numpy.concatenate(data)
        indices = jax.numpy.concatenate(indices, axis=0)
        
        return jsp.BCOO((data, indices), shape=shape)
    
    @staticmethod
    def special_kv(v, z):
        eps = 1e-10
        result = jax.numpy.exp(-z) * (1 + v / (2 * z + eps))
        return result
    
    @staticmethod
    def asformat(matrix, format):
        """
        Converts a sparse matrix to a desired format.

        If the input is a scipy sparse matrix (e.g., 'csc', 'csr'), returns the input matrix itself.
        If the input is a JAX BCOO matrix, processes it according to the specified format.
        
        Parameters
        ----------
        matrix : scipy sparse matrix or jsp.BCOO
            The input sparse matrix.
        format : str
            Desired sparse format ('dense', 'bcoo').
        
        Returns
        -------
        Processed matrix in the desired format.
        """
        if isinstance(matrix, (scipy.sparse.csc_matrix, scipy.sparse.coo_matrix, scipy.sparse.dia_matrix,  scipy.sparse.dok_matrix)):
            return matrix
        if isinstance(matrix, jsp.BCOO):
            if format == 'dense':
                return matrix.todense()
            elif format == 'bcoo':
                return matrix
            else:
                raise ValueError(f"JAX backend does not support format '{format}'. Only 'dense' and 'bcoo' are supported.")
        
        # If the input type is unexpected, raise an error
        raise TypeError(f"Unsupported matrix type: {type(matrix)}")
    
    @staticmethod
    def speye(n, m=None, k=0, dtype=jax.numpy.float32):
        """
        在 JAX 中创建一个稀疏单位矩阵，类似于 SciPy 的 sparse.eye。

        Parameters
        ----------
        n : int
            矩阵的行数。
        m : int, optional
            矩阵的列数，如果未指定，默认为 n (生成方阵)。
        k : int, optional
            主对角线的偏移量，默认为 0。
        dtype : data-type, optional
            数据类型，默认为 jax.numpy.float32。

        Returns
        -------
        BCOO 稀疏矩阵
            返回一个 n x m 的稀疏单位矩阵。
        """
        if m is None:
            m = n

        # 确定对角线元素的数量
        if k >= 0:
            diag_size = min(n, m - k)
            row_indices = jax.numpy.arange(diag_size)
            col_indices = row_indices + k
        else:
            diag_size = min(n + k, m)
            row_indices = jax.numpy.arange(diag_size) - k
            col_indices = jax.numpy.arange(diag_size)

        # 构造对角线上非零元素
        data = jax.numpy.ones(diag_size, dtype=dtype)

        # 构造 COO 格式的索引
        indices = jax.numpy.stack([row_indices, col_indices], axis=1)

        # 创建稀疏矩阵
        shape = (n, m)
        eye_sparse = jsp.BCOO((data, indices), shape=shape)

        return eye_sparse
    
    @staticmethod
    def speigsh(A, k=6, max_iter=100):
        return krylov_eigsh_jax(A, k, max_iter)


@custom_vjp
def pbdv_jax(n, x):
    return pbdv(n, x)[0]

def pbdv_jax_fwd(n, x):
    res = pbdv_jax(n, x)
    return res, (res, n, x)

def pbdv_jax_bwd(res, g):
    res, n, x = res
    eps = 1e-5
    dres_dx = (pbdv(n, x + eps)[0] - pbdv(n, x - eps)[0]) / (2 * eps)
    return (None, g * dres_dx)

pbdv_jax.defvjp(pbdv_jax_fwd, pbdv_jax_bwd)

def initialize_jax_backend():
    JaxBackend.pbdv_jax = staticmethod(pbdv_jax)

def backend_dependent_vjp(fn):
    if backend.__name__ == 'jax':
        return jax.custom_vjp(fn)
    else:
        return fn

@jit
def krylov_expm_jax(A, v, m):
    """
    使用 Krylov 子空间方法在 JAX 中计算矩阵的指数函数的近似值。
    
    Parameters:
    -----------
    A : array
        要计算指数函数的矩阵，必须是 JAX ndarray。
    v : array
        Krylov 子空间生成的初始向量，必须是 JAX ndarray。
    m : int
        Krylov 子空间的维数。
        
    Returns:
    --------
    expm_A_v : array
        近似计算出的矩阵指数作用在向量 v 上的结果。
    """
    n = A.shape[0]
    V = jax.numpy.zeros((n, m))
    H = jax.numpy.zeros((m, m))
    beta = jax.numpy.linalg.norm(v)
    V = V.at[:, 0].set(v / beta)

    for j in range(m - 1):
        w = A @ V[:, j]
        for i in range(j + 1):
            H = H.at[i, j].set(jax.numpy.dot(V[:, i].conj(), w))
            w -= H[i, j] * V[:, i]
        H = H.at[j + 1, j].set(jax.numpy.linalg.norm(w))
        if H[j + 1, j] > 1e-10:
            V = V.at[:, j + 1].set(w / H[j + 1, j])

    # 计算 H 的指数矩阵 (这里使用 jax.scipy.linalg.expm)
    expH = jax.scipy.linalg.expm(H)

    # 返回矩阵指数与向量 v 的乘积
    return beta * (V @ expH[:, 0])

@jax.jit
def krylov_eigsh_jax(A, k=6, max_iter=100):
    """
    使用 Krylov 子空间方法在 JAX 中计算前 k 个特征值和特征向量。

    Parameters:
    -----------
    A : array
        要计算特征值和特征向量的对称矩阵，必须是 JAX ndarray。
    k : int
        计算前 k 个特征值和特征向量。
    max_iter : int
        Krylov 子空间迭代的最大次数。

    Returns:
    --------
    evals : array
        前 k 个特征值。
    evecs : array
        前 k 个特征向量。
    """
    n = A.shape[0]
    V = jax.numpy.zeros((n, k))
    H = jax.numpy.zeros((k, k))

    v = jax.random.normal(jax.random.PRNGKey(0), (n,))
    v = v / jax.linalg.norm(v)
    beta = 0

    for j in range(k):
        w = A @ v
        alpha = jax.numpy.dot(w, v)
        w = w - beta * V[:, j - 1] if j > 0 else w
        w = w - alpha * v
        beta = jax.linalg.norm(w)

        V = V.at[:, j].set(v)
        H = H.at[j, j].set(alpha)

        if j + 1 < k:
            v = w / beta
            H = H.at[j + 1, j].set(beta)
            H = H.at[j, j + 1].set(beta)

    # 计算 H 的特征值和特征向量
    evals, evecs = jax.numpy.linalg.eigh(H)
    evecs_full = V @ evecs

    return evals[:k], evecs_full[:, :k]




backend = NumpyBackend()

def set_backend(name):
    if name == 'numpy':
        backend.__class__ = NumpyBackend
    elif name == 'jax':
        backend.__class__ = JaxBackend
        initialize_jax_backend()
    else:
        raise ValueError(f"Unknown backend '{name}'")

