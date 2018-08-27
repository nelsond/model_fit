import numpy as np
import numpy.fft as fft
from .generator import Generator


def _argmax(xdata, ydata):
    """Returns point in xdata where ydata is maximum"""

    idx = np.argmax(ydata)
    return xdata[idx]


def _argmin(xdata, ydata):
    """Returns point in xdata where ydata is minimum"""

    idx = np.argmin(ydata)
    return xdata[idx]


def _max(_, ydata):
    """Returns point where ydata is maximum"""

    return np.max(ydata)


def _min(_, ydata):
    """Returns point where ydata is minimum"""

    return np.min(ydata)


def _pktopk(_, ydata):
    """Returns difference between maximum and minimum of ydata"""

    return np.max(ydata) - np.min(ydata)


def _mean(_, ydata):
    """Returns mean of ydata"""

    return np.mean(ydata)


def _fftfreq(xdata, ydata):
    """
    Returns major frequency component of (xdata, ydata) using
    (real-valued) fourier transform of ydata. Requires
    nearly-equal spacing of xdata.
    """

    dx = np.mean(xdata[1:] - xdata[:-1])
    freq = np.abs(fft.rfftfreq(xdata.size, d=dx))
    amp = np.abs(fft.rfft(ydata - np.mean(ydata)))
    idx = np.argmax(amp)

    return freq[idx]


def _firstmoment(xdata, ydata):
    """
    Returns 1st order moment (distribution center for Gaussian).
    """
    return np.sum(xdata*ydata)/np.sum(ydata)


def _secondmoment(xdata, ydata):
    """
    Returns 2nd order moment of distribution (variance for Gaussian).
    """
    center = _firstmoment(xdata, ydata)
    return np.abs(np.sum((xdata-center)**2 * ydata)/np.sum(ydata))


argmax = Generator(_argmax)
argmin = Generator(_argmin)
max = Generator(_max)
min = Generator(_min)
pktopk = Generator(_pktopk)
mean = Generator(_mean)
fftfreq = Generator(_fftfreq)
firstmoment = Generator(_firstmoment)
secondmoment = Generator(_secondmoment)
