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
    csc_matrix = staticmethod(scipy.sparse.csc_matrix)
    dia_matrix = staticmethod(scipy.sparse.dia_matrix)
    coo_matrix = staticmethod(scipy.sparse.coo_matrix)
    spidentity = staticmethod(scipy.sparse.identity)
    spkron = staticmethod(scipy.sparse.kron)
    expm = staticmethod(scipy.linalg.expm)
    spexpm = staticmethod(scipy.sparse.linalg.expm)

    @staticmethod
    def solve_csc_matrix(matrix):
        return matrix.tocsc()

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
    csc_matrix = staticmethod(jsp.BCOO)
    coo_matrix = staticmethod(jsp.BCOO)
    identy = staticmethod(jsp.eye)
    scipy = staticmethod(jax.scipy)
    grad = staticmethod(jax.grad)
    value_and_grad = staticmethod(jax.value_and_grad)
    eigh = staticmethod(jax.scipy.linalg.eigh)
    eigvalsh = staticmethod(jax.scipy.linalg.eigh)
    expm = staticmethod(jax.scipy.linalg.expm)
    
    @staticmethod
    def spidentity(n,format):
         # 构造单位矩阵的非零元素（对角线上的元素全为1）
        data = jax.numpy.ones(n)
        indices = jax.numpy.arange(n)[:, None]  # 行和列的索引是相同的
        indices = jax.numpy.hstack([indices, indices])  # 构造 (row, col) 索引对

        # 创建 BCOO 稀疏单位矩阵
        return jsp.BCOO((data, indices), shape=(n, n))

    @staticmethod
    def spkron(a: jsp.BCOO, b: jsp.BCOO, format):
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
        if isinstance(matrix, jsp.BCOO):
            return True
        else:
            return False

    @staticmethod
    def to_dense(matrix):
        if isinstance(matrix, jsp.BCOO):
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
    def dia_matrix(diagonal, shape):
        """
        在 JAX 中构建对角稀疏矩阵，类似于 SciPy 的 sparse.dia_matrix。
        
        Parameters
        ----------
        diagonal : array
            对角线上非零元素的值。
        shape : tuple
            稀疏矩阵的形状。
        
        Returns
        -------
        BCOO 稀疏矩阵
        """
        # 非零元素为对角线上的元素
        data = jax.numpy.array(diagonal)

        # 对角线元素的位置 (row, col)
        indices = jax.numpy.arange(len(diagonal))
        indices = jax.numpy.stack([indices, indices], axis=1)  # 构造 (row, col) 索引对

        # 构建稀疏 BCOO 矩阵
        return jsp.BCOO((data, indices), shape=shape)
    
    @staticmethod
    def solve_csc_matrix(matrix):
        coo_matrix = matrix.tocoo()  # 先将其转换为 COO 格式
        data = jax.numpy.array(coo_matrix.data)
        row = jax.numpy.array(coo_matrix.row)
        col = jax.numpy.array(coo_matrix.col)
        indices = jax.numpy.stack([row, col], axis=1)
        shape = coo_matrix.shape
        return jsp.BCOO((data, indices), shape=shape)

    @staticmethod
    def spexpm(A):
        v = jax.numpy.ones(A.shape[0])  # 这里使用一个全1的单位向量作为 Krylov 子空间的初始向量
        m = 10 
        return krylov_expm_jax(A,v,m)

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




backend = NumpyBackend()

def set_backend(name):
    if name == 'numpy':
        backend.__class__ = NumpyBackend
    elif name == 'jax':
        backend.__class__ = JaxBackend
        initialize_jax_backend()
    else:
        raise ValueError(f"Unknown backend '{name}'")

