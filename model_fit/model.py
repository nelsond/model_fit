import inspect
from scipy.optimize import curve_fit
import numpy as np


class Model:
    _HTML_ROW_TEMPLATE = """
    <tr>
        <td style="text-align:center;">
            {}
        </td>
        <td style="text-align:right;">
            <pre>{:f}</pre>
        </td>
       <td style="text-align:right;">
            <pre>{:f}</pre>
        </td>
    </tr>""".strip()

    def __init__(self, func, params=None, guess_gen=None):
        """Creates a new fit model.

        Args:
            func (function): model function to fit
            params (list): optional names of parameters, argument names of func
                           are used otherwise
            guess_gen (list or dict): list or dict with functions or fixed
                                      values for generating guesses

        Returns:
            Model
        """

        self._fit_func = func
        self._param_names = params or self._guess_param_names(func)

        if type(guess_gen) == dict:
            self._guess_gen = [guess_gen.get(p, 1) for p in self._param_names]
        else:
            self._guess_gen = guess_gen

        self._has_fitted = False

    def fit(self, xdata, ydata, guess=None, **kwargs):
        """Fits data to model and returns optimized parameters

        Args:
            xdata (array): x data values
            ydata (array): y data values
            guess (list): optional guess (otherwise guess generator is used)
            kwargs (dict): keyword arguments passed to
                           scipy.optimize.curve_fit

        Returns:
            None
        """

        self._xdata = np.array(xdata)
        self._ydata = np.array(ydata)

        self._params, self._cov = curve_fit(self._fit_func,
                                            self._xdata,
                                            self._ydata,
                                            p0=(guess or self._guess),
                                            **kwargs)

        self._has_fitted = True

    def plot(self, ax=None, n=None, dx=None, style='-', **kwargs):
        """Plots fitted function with xdata passed to Model.fit from min(xdata)
        to max(xdata)

        Args:
            n (int): optional number of points used for plot
            dx (float): optional spacing of points used for plot
            style (string): optional linestyle and markerstyle for
                            matplotlib.pyplot.plot
            kwargs (dict): keyword arguments passed to matplotlib.pyplot.plot

        Returns:
            None
        """
        import matplotlib.pyplot as plt

        if not self._has_fitted:
            return

        xmin, xmax = min(self._xdata), max(self._xdata)

        if n is not None:
            xx = np.linspace(xmin, xmax, n)
        elif dx is not None:
            xx = np.arange(xmin, xmax, dx)
        else:
            xx = self._xdata

        if ax is None:
            ax = plt.gca()

        ax.plot(xx, self.fitted_func(xx), style, **kwargs)

    def display_html(self):
        """Shows HTML table of fit params in jupyter notebook environment

        Returns:
            None
        """

        try:
            from IPython.core.display import display
            display(self)

        except ImportError:
            print('Cannot import IPython.core.display module. '
                  'Make sure you run this function in an '
                  'ipython/jupyter notebook.')

    @property
    def func(self):
        """Model function

        Returns:
            function
        """

        return self._fit_func

    @property
    def params(self):
        """Optimal values for parameters (returned from
           scipy.optmize.curve_fit)

        Returns:
            dict
        """

        if self._has_fitted is False:
            return None

        return dict(zip(self._param_names, self._params))

    @property
    def covariance(self):
        """Covariance of fitted parameters (returned from
           scipy.optimize.curve_fit)

        Returns:
            2d array
        """

        if self._has_fitted is False:
            return None

        return self._cov

    @property
    def param_errors(self):
        """One standard deviation error on parameters calculated from
           covariance

        Returns:
            dict
        """

        if self._has_fitted is False:
            return None

        diag = np.diag(self.covariance)
        err = np.sqrt(diag)

        return dict(zip(self._param_names, err))

    @property
    def residuals(self):
        """ residuals of the fit

        Returns:
            array
        """

        if self._has_fitted is False:
            return None

        return self._ydata - self.fitted_func(self._xdata)

    @property
    def fitted_func(self):
        """Returns fit function with optimal parameter values

        Returns:
            function
        """

        if self._has_fitted is False:
            return None

        params = list(self._params)
        return lambda x: self._fit_func(x, *params)

    @property
    def xdata(self):
        """x data points used for fit (copy of actual data)

        Returns:
            array
        """

        if self._has_fitted is False:
            return None

        return np.copy(self._xdata)

    @property
    def ydata(self):
        """y data points used for fit (copy of actual data)

        Returns:
            array
        """

        if self._has_fitted is False:
            return None

        return np.copy(self._ydata)

    @property
    def _guess(self):
        guess_gen = self._guess_gen

        if not guess_gen:
            return ([1] * len(self._param_names))
        else:
            return [self._gen_guess(g) for g in guess_gen]

    def _repr_html_(self):
        if self._has_fitted is False:
            return self.__repr__()

        params = self.params
        errors = self.param_errors

        html = """
        <table>
            <tr>
                <th style="text-align:center;">Parameter</th>
                <th style="text-align:center;">Optimal value</th>
                <th style="text-align:center;">1σ error</th>
            </tr>
        """.strip()
        for name in self._param_names:
            content = name, params[name], errors[name]
            html += self._HTML_ROW_TEMPLATE.format(*content)

        html += '</table>'

        return html

    def _gen_guess(self, g):
        if callable(g):
            return g(self._xdata, self._ydata)
        else:
            return g

    def __getitem__(self, key):
        if self._has_fitted is False:
            raise KeyError(key)

        return self.params[key]

    def __getattr__(self, name):
        if (self._has_fitted is False or
                name not in self._param_names):
            raise AttributeError(name)

        return self.params[name]

    @staticmethod
    def _guess_param_names(func):
        signature = inspect.signature(func)
        param_names = list(signature.parameters)
        param_names.pop(0)

        return param_names
