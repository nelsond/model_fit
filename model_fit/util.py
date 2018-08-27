from .model import Model


def fit_model(func, xdata, ydata, params=None, guess_gen=None, **kwargs):
    """Utitlity function that creates a Model and runs a fit

    Args:
        func (function): model function to fit
        xdata (array): x data values
        ydata (array): y data values
        params (list): otpional names of parameters, argument names of func are
                       used otherwise
        guess_gen (list or dict): list or dict with functions or fixed values
                                  for generating guesses, see fit.
        kwargs (dict): keyword arguments passed to Model.fit

    Returns:
        Model
    """

    model = Model(func, params, guess_gen)
    model.fit(xdata, ydata, **kwargs)

    return model
