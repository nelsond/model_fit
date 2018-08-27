class Generator:
    def __init__(self, func):
        """Creates a new guess generator.

        Args:
            func (function): guess generating function (must accept two
                             arguments: xdata and ydata)

        Returns:
            Generator
        """

        # extract func if Generator
        if (type(func) == Generator):
            self._func = func.func
        else:
            self._func = func

        # clone docstring of function
        self.__doc__ = func.__doc__

    @property
    def func(self):
        """Generator function.

        Returns:
            function
        """

        return self._func

    def __call__(self, xdata, ydata):
        return self._func(xdata, ydata)

    def _calculated(self, other, opfunc):
        def new_func(xdata, ydata):
            a = self(xdata, ydata) if callable(self) else self
            b = other(xdata, ydata) if callable(other) else other

            return opfunc(a, b)

        return Generator(new_func)

    def __add__(self, other):
        return self._calculated(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._calculated(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._calculated(other, lambda a, b: b - a)

    def __mul__(self, other):
        return self._calculated(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._calculated(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return self._calculated(other, lambda a, b: b / a)

    def __floordiv__(self, other):
        return self._calculated(other, lambda a, b: a // b)

    def __rfloordiv__(self, other):
        return self._calculated(other, lambda a, b: b // a)

    def __pow__(self, other):
        return Generator(lambda xdata, ydata: self._func(xdata, ydata)**other)
