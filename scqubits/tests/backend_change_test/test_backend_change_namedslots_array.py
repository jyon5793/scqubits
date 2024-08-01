import pytest
import numpy as np
import jax
from scqubits import backend_change
from scqubits.core.namedslots_array import NamedSlotsNdarray


@pytest.fixture
def namedslots_params():
    values_by_name = {
        'param1': np.array([1, 2, 3, 4]),
        'param2': np.array([5, 6, 7, 8])
    }
    input_array = np.random.rand(4, 4)
    return input_array, values_by_name


def test_namedslotsndarray_numpy(namedslots_params):
    backend_change.set_backend('numpy')
    input_array, values_by_name = namedslots_params

    # 创建 NamedSlotsNdarray 实例
    named_array = NamedSlotsNdarray(input_array, values_by_name)

    # 测试基本属性
    assert named_array.shape == input_array.shape
    assert np.allclose(named_array, input_array)

    # 测试切片和索引
    assert np.allclose(named_array[0], input_array[0])
    assert np.allclose(named_array[:, 1], input_array[:, 1])

    # 测试名称和值
    assert named_array._parameters.paramvals_by_name == values_by_name


def test_namedslotsndarray_jax(namedslots_params):
    backend_change.set_backend('jax')
    input_array, values_by_name = namedslots_params

    # 创建 NamedSlotsNdarray 实例
    named_array = NamedSlotsNdarray(input_array, values_by_name)

    # 测试基本属性
    assert named_array.shape == input_array.shape
    assert np.allclose(np.array(named_array), input_array)

    # 测试切片和索引
    assert np.allclose(np.array(named_array[0]), input_array[0])
    assert np.allclose(np.array(named_array[:, 1]), input_array[:, 1])

    # 测试名称和值
    assert named_array._parameters.paramvals_by_name == values_by_name


def test_backend_switch_between_numpy_and_jax(namedslots_params):
    # 切换到 numpy 后端
    backend_change.set_backend('numpy')
    input_array, values_by_name = namedslots_params
    named_array_numpy = NamedSlotsNdarray(input_array, values_by_name)

    # 切换到 jax 后端
    backend_change.set_backend('jax')
    named_array_jax = NamedSlotsNdarray(input_array, values_by_name)

    # 比较结果
    assert named_array_numpy.shape == named_array_jax.shape
    assert np.allclose(np.array(named_array_numpy), np.array(named_array_jax))

    # 测试切片和索引
    assert np.allclose(np.array(named_array_numpy[0]), np.array(named_array_jax[0]))
    assert np.allclose(np.array(named_array_numpy[:, 1]), np.array(named_array_jax[:, 1]))


if __name__ == "__main__":
    pytest.main()