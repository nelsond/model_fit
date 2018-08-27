from model_fit.guess import Generator

import numpy as np
from pytest import fixture


@fixture
def xmax_generator():
    def func(x, y):
        return np.max(x)

    return Generator(func)


# __init__
def test_init_sets_function():
    def func(x, y):
        return x[0]

    g = Generator(func)
    assert(g._func == func)


def test_init_steals_func_docstring():
    def func(x, y):
        """Some docstring"""
        return x[0]

    g = Generator(func)
    assert(g.__doc__ == func.__doc__)


def test_init_extracts_func_from_generator_argument():
    def func(x, y):
        return x[0]

    g1 = Generator(func)
    g2 = Generator(g1)
    assert(g1.func == g2.func)


# func
def test_func_returns_function():
    def func(x, y):
        return [0]

    g = Generator(func)
    assert(g.func == func)


# __call__
def test_call_calls_function():
    def func(x, y):
        return x[0]
    g = Generator(func)
    result = g(np.array([1]), np.array([1]))
    assert(result == 1)


# __add__
def test_adding_two_generators(xmax_generator):
    new_generator = xmax_generator + xmax_generator
    x = np.array([1])
    y = np.array([0])

    print(new_generator)

    assert(new_generator(x, y) == 2)


def test_adding_a_number_to_a_generator(xmax_generator):
    new_generator = xmax_generator + 10
    x = np.array([1])
    y = np.array([0])

    assert(new_generator(x, y) == 11)


# __radd__
def test_adding_a_generator_to_a_number(xmax_generator):
    new_generator = 10 + xmax_generator
    x = np.array([1])
    y = np.array([0])

    assert(new_generator(x, y) == 11)


# __sub__
def test_subtracting_two_generators(xmax_generator):
    new_generator = xmax_generator - xmax_generator
    x = np.array([1])
    y = np.array([0])

    assert(new_generator(x, y) == 0)


def test_subtracting_a_number_from_a_generator(xmax_generator):
    new_generator = xmax_generator - 10
    x = np.array([1])
    y = np.array([0])

    assert(new_generator(x, y) == -9)


# __rsub__
def test_subtraction_a_generator_from_a_number(xmax_generator):
    new_generator = 10 - xmax_generator
    x = np.array([1])
    y = np.array([0])

    assert(new_generator(x, y) == 9)


# __mul__
def test_multiplying_two_generators(xmax_generator):
    new_generator = xmax_generator * xmax_generator
    x = np.array([2])
    y = np.array([0])

    assert(new_generator(x, y) == 4)


def test_multiplying_a_generator_with_a_number(xmax_generator):
    new_generator = xmax_generator * 10
    x = np.array([1])
    y = np.array([0])

    assert(new_generator(x, y) == 10)


# __rmul__
def test_multiplying_a_number_with_a_generator(xmax_generator):
    new_generator = 20 * xmax_generator
    x = np.array([1])
    y = np.array([0])

    assert(new_generator(x, y) == 20)


# __truediv__
def test_dividing_two_generators(xmax_generator):
    new_generator = xmax_generator / xmax_generator
    x = np.array([2])
    y = np.array([0])

    assert(new_generator(x, y) == 1)


def test_dividing_a_generator_by_a_number(xmax_generator):
    new_generator = xmax_generator / 3
    x = np.array([2])
    y = np.array([0])

    assert(new_generator(x, y) == (2/3))


# __rtruediv__
def test_dividing_a_number_by_a_generator(xmax_generator):
    new_generator = 1 / xmax_generator
    x = np.array([2])
    y = np.array([0])

    assert(new_generator(x, y) == (1/2))


# __floordiv__
def test_floor_dividing_two_generators(xmax_generator):
    new_generator = xmax_generator // xmax_generator
    x = np.array([2])
    y = np.array([0])

    assert(new_generator(x, y) == 1)


def test_floor_dividing_a_generator_by_a_number(xmax_generator):
    new_generator = xmax_generator // 3
    x = np.array([2])
    y = np.array([0])

    assert(new_generator(x, y) == (2//3))


# __rfloordiv__
def test_floor_dividing_a_number_by_a_generator(xmax_generator):
    new_generator = 1 // xmax_generator
    x = np.array([2])
    y = np.array([0])

    assert(new_generator(x, y) == (1//2))


# __pow__
def test_pow_generator(xmax_generator):
    new_generator = xmax_generator**3
    x = np.array([2])
    y = np.array([0])

    assert(new_generator(x, y) == 2**3)
