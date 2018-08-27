from model_fit import Model

import pytest
import numpy as np
import matplotlib.pyplot as plt


@pytest.fixture
def fit_data():
    xx = np.array([-3., -2., -1., 0., 1., 2., 3.])
    yy = xx**2

    xx += np.random.rand(xx.size) * 0.01
    yy += np.random.rand(xx.size) * 0.01

    return (xx, yy)


@pytest.fixture
def fit_func():
    return lambda x, A, x0, c: A * ((x-x0)**2) + c


# .__init__
def test_init_guesses_param_names(fit_func):
    model = Model(fit_func)
    assert(model._param_names == ['A', 'x0', 'c'])


def test_init_accepts_custom_param_names(fit_func):
    model = Model(fit_func, params=['a', 'b', 'C'])
    assert(model._param_names == ['a', 'b', 'C'])


def test_init_accepts_guess_gen_dict(fit_func):
    model = Model(fit_func, guess_gen={'A': 10})
    assert(model._guess_gen == [10, 1, 1])


def test_init_accepts_guess_gen_list(fit_func):
    model = Model(fit_func, guess_gen=[10, 10, 10])
    assert(model._guess_gen == [10, 10, 10])


# .fit
def test_fit_does_fit(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)
    assert(model._has_fitted)


def test_fit_does_fit_with_guess_from_generator(fit_func, fit_data):
    model = Model(fit_func, guess_gen={'A': lambda x, y: 10})
    model.fit(*fit_data)
    assert(model._guess == [10, 1, 1])


def test_fit_converts_data_to_numpy_array(fit_func):
    model = Model(fit_func)
    model.fit([-3, -2, 0, 2, 3], [5, 4, 0, 4, 5])

    assert(type(model._xdata) == np.ndarray)
    assert(type(model._ydata) == np.ndarray)


# .plot
def test_plot_does_nothing_before_fit(fit_func, fit_data):
    model = Model(fit_func)

    plt.figure()
    model.plot()

    assert(len(plt.gca().lines) == 0)


def test_plot_plots_fitted_func(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)

    plt.figure()
    model.plot()

    assert(len(plt.gca().lines) > 0)


def test_plot_uses_n_points(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)

    plt.figure()
    model.plot(n=1000)

    line = plt.gca().lines[0]
    xdata = line.get_xdata()

    assert(len(xdata) == 1000)


def test_plot_uses_dx_point_spacing(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)

    plt.figure()
    model.plot(dx=0.1)

    line = plt.gca().lines[0]
    xdata = line.get_xdata()

    assert(len(xdata) == 60 or len(xdata) == 61)


def test_plot_uses_plot_style(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)

    plt.figure()
    model.plot(style='--')

    line = plt.gca().lines[0]
    style = line.get_linestyle()

    assert(style == '--')


# .display_html
def test_does_not_raise_outside_of_jupyter_notebook(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)

    model.display_html()


# ._repr_html_
def test_repr_html_is_repr_before_fit(fit_func):
    model = Model(fit_func)
    assert(model._repr_html_() == model.__repr__())


def test_repr_html_is_html_table_after_fit(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)
    assert('<table>' in model._repr_html_())


# .__getitem__
def test_getitem_raises_key_error_before_fit(fit_func):
    model = Model(fit_func)

    with pytest.raises(KeyError):
        model['A']


def test_getitem_returns_fit_param_after_fit(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)
    assert(isinstance(model['A'], float))


# .__getattr__
def test_getattr_raises_attribute_error_before_fit(fit_func):
    model = Model(fit_func)

    with pytest.raises(AttributeError):
        model.A


def test_getattr_returns_fit_param_after_fit(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)
    assert(isinstance(model.A, float))


# .func (property)
def test_func_is_model_function(fit_func):
    model = Model(fit_func)
    assert(model.func == fit_func)


# .fitted_func (property)
def test_fitted_func_is_none_before_fit(fit_func):
    model = Model(fit_func)
    assert(model.fitted_func is None)


def test_fitted_func_is_function_after_fit(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)
    assert(callable(model.fitted_func))


# .params (property)
def test_params_is_none_before_fit(fit_func):
    model = Model(fit_func)
    assert(model.params is None)


def test_params_is_dict_with_optimal_parameters_after_fit(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)
    assert(sorted(model.params.keys()) == ['A', 'c', 'x0'])


# .param_errors (property)
def test_param_errors_is_none_before_fit(fit_func):
    model = Model(fit_func)
    assert(model.param_errors is None)


def test_param_errors_is_dict_with_errors_after_fit(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)
    err = np.array([model.param_errors[n] for n in model._param_names])

    expected_err = np.sqrt(np.diag(model.covariance))
    assert(np.array_equal(err, expected_err))


# .residuals (property)
def test_residuals_are_none_before_fit(fit_func):
    model = Model(fit_func)
    assert (model.residuals is None)


def test_resiuduals_is_array_with_residuals_after_fit(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)

    expected_residuals = fit_data[1] - model.fitted_func(fit_data[0])
    assert(np.array_equal(model.residuals, expected_residuals))


# .covariance (property)
def test_covariance_is_none_before_fit(fit_func):
    model = Model(fit_func)
    assert(model.covariance is None)


def test_covariance_is_internal_cov_after_fit(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)
    assert(np.array_equal(model.covariance, model._cov))


# .xdata (property)
def test_xdata_is_none_before_fit(fit_func):
    model = Model(fit_func)
    assert(model.xdata is None)


def test_xdata_is_private_xdata_after_fit(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)
    assert(np.array_equal(model.xdata, model._xdata))


# .ydata (property)
def test_ydata_is_none_before_fit(fit_func):
    model = Model(fit_func)
    assert(model.ydata is None)


def test_ydata_is_private_ydata_after_fit(fit_func, fit_data):
    model = Model(fit_func)
    model.fit(*fit_data)
    assert(np.array_equal(model.ydata, model._ydata))
