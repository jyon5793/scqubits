import pytest
import numpy as np
import jax.numpy as jnp
from scqubits import backend_change as bc
from scqubits.core.namedslots_array import NamedSlotsNdarray
from scqubits.core.hilbert_space import HilbertSpace
from scqubits.core.circuit_routines import CircuitRoutines
from scqubits import backend_change
from scipy.sparse import csc_matrix
import jax
import sympy as sm

def test_identity_wrap_for_hd_jax():
    # 切换到 JAX 后端
    backend_change.set_backend('jax')

    # 示例输入数据
    jax_array = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # 用于测试的 JAX 数组
    sparse_matrix = csc_matrix(jax_array)  # 稀疏矩阵作为输入
    child_instance = "subsystem_example"  # 模拟一个子系统实例
    bare_esys = None  # 测试时可以选择不提供
    
    # 创建 CircuitRoutines 实例
    circuit_routines_instance = CircuitRoutines()
    
    # 手动设置缺失的属性
    object.__setattr__(circuit_routines_instance, "_frozen", False)
    circuit_routines_instance.hierarchical_diagonalization = False  # 假设为 False
    circuit_routines_instance.subsystems = []  # 假设 subsystems 是空的

    # 前向传播测试
    result = circuit_routines_instance.identity_wrap_for_hd(sparse_matrix, child_instance, bare_esys)
    print("Forward result:", result)

    # 定义目标函数，测试反向传播
    def func_to_differentiate(operator):
        return jnp.sum(circuit_routines_instance.identity_wrap_for_hd(operator, child_instance, bare_esys).full())

    # 计算梯度
    grad_func = jax.grad(func_to_differentiate)
    # grad_result = grad_func(sparse_matrix)
    
    # print("Gradient result:", grad_result)

    # # 断言，确保结果的正确性
    # assert isinstance(grad_result, jnp.ndarray), "Gradient should be a JAX array."
    # print("Test passed.")

# x, y, z = sm.symbols('x y z')

# # 定义哈密顿量 H_sys，包含 x, y
# H_sys = x**2 + y**2

# # 定义常数项表达式 constants_expr，包含 x, y, z
# constants_expr = x**2 + 2*y + z

# # 2. 定义测试函数，将 SymPy 表达式转换为 JAX 兼容的数值函数
# def test_constants_in_subsys(x_val, y_val, z_val):
#     backend_change.set_backend('jax')

#     # 示例输入数据
#     jax_array = jnp.array([[1.0, 2.0], [3.0, 4.0]])  # 用于测试的 JAX 数组
#     sparse_matrix = csc_matrix(jax_array)  # 稀疏矩阵作为输入
#     child_instance = "subsystem_example"  # 模拟一个子系统实例
#     bare_esys = None  # 测试时可以选择不提供
    
    # 创建 CircuitRoutines 实例
    circuit_routines_instance = CircuitRoutines()
    
    # 手动设置缺失的属性
    object.__setattr__(circuit_routines_instance, "_frozen", False)
    circuit_routines_instance.hierarchical_diagonalization = False  # 假设为 False
    circuit_routines_instance.subsystems = []  # 假设 subsystems 是空的
#     # 调用自定义的函数，传入具体数值
#     result = circuit_routines_instance._constants_in_subsys(H_sys, constants_expr)
    
#     # 使用 SymPy lambdify 将符号表达式转换为可计算的数值函数
#     H_sys_func = sm.lambdify([x, y], H_sys, modules='numpy')
#     constants_expr_func = sm.lambdify([x, y, z], constants_expr, modules='numpy')
    
#     # 计算前向传播值
#     H_sys_val = H_sys_func(x_val, y_val)
#     constants_expr_val = constants_expr_func(x_val, y_val, z_val)
    
#     # 返回结果
#     return result, H_sys_val, constants_expr_val

# # 3. 测试梯度

# # 定义一个简单的 JAX 函数，封装 _constants_in_subsys 并返回标量
# def jax_test_function(x_val, y_val, z_val):
#     # 将 SymPy 表达式转换为数值函数
#     result, _, _ = test_constants_in_subsys(x_val, y_val, z_val)
#     return jnp.array(result)  # 转换为 JAX 数组

# # 4. 计算梯度
# x_val, y_val, z_val = 1.0, 2.0, 3.0

# # 使用 JAX grad 函数计算梯度
# def test():
#     grad_x = jax.grad(jax_test_function, argnums=0)(x_val, y_val, z_val)
#     grad_y = jax.grad(jax_test_function, argnums=1)(x_val, y_val, z_val)
#     grad_z = jax.grad(jax_test_function, argnums=2)(x_val, y_val, z_val)

#     # 5. 打印结果
#     print(f"Grad wrt x: {grad_x}")
#     print(f"Grad wrt y: {grad_y}")
#     print(f"Grad wrt z: {grad_z}")
#     assert(False)

def mixed_test():
    hamiltonian_expr = sm.sympify("Q1**2 + θ1**2 + Q2**2 + θ2**2")
    transformation_matrix = np.array([[1.0, 0.0], [0.0, 1.0]])

    # 2. 定义变换矩阵
    transformation_matrix = np.array([[1, 0.5], [-0.5, 1]])

    # 3. 创建测试实例
    backend_change.set_backend('jax')
    circuit_routines_instance = CircuitRoutines()

    # 手动设置缺失的属性
    object.__setattr__(circuit_routines_instance, "_frozen", False)
    circuit_routines_instance.hierarchical_diagonalization = False  # 假设为 False
    circuit_routines_instance.subsystems = []  # 假设 subsystems 是空的
    circuit_routines_instance.var_categories = {"extended": [1, 2]}

    # 4. 测试函数：定义 JAX 包装的计算哈密顿量的函数
    def jax_test_function(transformation_matrix):
        return circuit_routines_instance._transform_hamiltonian(hamiltonian_expr, transformation_matrix)

    # 5. 计算梯度
    grad_matrix = jax.grad(jax_test_function)(transformation_matrix)

    # 6. 输出梯度
    print("Transformation matrix gradient:")
    print(grad_matrix)

    # 7. 测试前向传播结果
    H_transformed = circuit_routines_instance._transform_hamiltonian(hamiltonian_expr, transformation_matrix)
    print("Transformed Hamiltonian:")
    print(H_transformed)

def init_object():
    backend_change.set_backend('jax')
    circuit_routines_instance = CircuitRoutines()

    # 手动设置缺失的属性
    object.__setattr__(circuit_routines_instance, "_frozen", False)
    circuit_routines_instance.hierarchical_diagonalization = False  # 假设为 False
    circuit_routines_instance.subsystems = []  # 假设 subsystems 是空的
    circuit_routines_instance.var_categories = {"extended": [1, 2]}
    return circuit_routines_instance

def csc_matrix_test():
    circuit_routines_instance = init_object()
    hamiltonian_matrix = circuit_routines_instance._evaluate_hamiltonian()

    # 定义一个简单的 JAX 函数来计算梯度
    def jax_test_function():
        return circuit_routines_instance._evaluate_hamiltonian()

    # 计算梯度
    grad_matrix = jax.grad(jax_test_function)()

    # 输出梯度
    print("Gradient of the Hamiltonian matrix:")
    print(grad_matrix)
    assert(False)

def csc_matrix_test2():
    circuit_routines_instance = init_object()
    def jax_test_function():
        return circuit_routines_instance.hamiltonian()

    # 计算梯度
    grad_matrix = jax.grad(jax_test_function)()

    # 输出梯度
    print("Gradient of the Hamiltonian matrix:")
    print(grad_matrix)

    # 测试前向传播结果
    H_transformed = circuit_routines_instance.hamiltonian()
    print("Hamiltonian:")
    print(H_transformed)