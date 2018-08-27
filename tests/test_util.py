from model_fit import fit_model
from model_fit import Model

from pytest import fixture
import numpy as np


@fixture
def fit_data():
    xx = np.array([-3., -2., -1., 0., 1., 2., 3.])
    yy = xx**2

    xx += np.random.rand(xx.size) * 0.01
    yy += np.random.rand(xx.size) * 0.01

    return (xx, yy)


@fixture
def fit_func():
    return lambda x, A, x0, c: A * ((x-x0)**2) + c


# model_fit
def test_model_fit_returns_model(fit_func, fit_data):
    model = fit_model(fit_func, *fit_data)

    assert(type(model) == Model)


def test_model_fit_returns_fitted_model(fit_func, fit_data):
    model = fit_model(fit_func, *fit_data)

    assert(model.params)
