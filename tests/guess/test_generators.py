import model_fit.guess as g

import pytest
import numpy as np


# argmax
def test_argmax():
    x = np.array([1, 2, 3])
    y = np.array([8, -1, 100])
    assert(g.argmax(x, y) == 3)


# argmin
def test_argmin():
    x = np.array([1, 2, 3])
    y = np.array([8, -1, 100])
    assert(g.argmin(x, y) == 2)


# max
def test_max():
    x = np.array([1, 2, 3])
    y = np.array([8, -1, 100])
    assert(g.max(x, y) == 100)


# min
def test_min():
    x = np.array([1, 2, 3])
    y = np.array([-8, 1, -100])
    assert(g.min(x, y) == -100)


# pktopk
def test_pktopk():
    x = np.array([1, 2, 3])
    y = np.array([-1, 10, 0])
    assert(g.pktopk(x, y) == 11)


# mean
def test_mean():
    x = np.array([1, 2, 3])
    y = np.array([0, 1, 2])
    assert(g.mean(x, y) == 1)


# fftfreq
def test_fftfreq():
    f = 0.1
    x = np.arange(0, 10)
    y = np.sin(2*np.pi*f*x)
    assert(g.fftfreq(x, y) == f)


# firstmoment
def test_firstmoment():
    f = 2
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([1, 1, 2, 1, 1])
    assert(g.firstmoment(x, y) == f)


# secondmoment
def test_secondmoment():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([1, 1, 2, 1, 1])
    secondmoment = (y*(x-2)**2) / np.sum(y)

    assert(pytest.approx(g.secondmoment(x, y), 1.6667))
