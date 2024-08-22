import numpy as np
import jax
from functools import wraps
from scipy import sparse
from jax import custom_vjp
from scipy.special import pbdv


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

class JaxBackend(Backend):
    __name__ = 'jax'
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

    scipy = staticmethod(jax.scipy)
    grad = staticmethod(jax.grad)
    value_and_grad = staticmethod(jax.value_and_grad)
    eigh = staticmethod(jax.scipy.linalg.eigh)

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


backend = NumpyBackend()

def set_backend(name):
    if name == 'numpy':
        backend.__class__ = NumpyBackend
    elif name == 'jax':
        backend.__class__ = JaxBackend
        initialize_jax_backend()
    else:
        raise ValueError(f"Unknown backend '{name}'")

