"""The main example of the paper: N = 3, eps = 1/4, six-piece local map f (numba versions, for the numerics).

f is the piecewise linear interpolation of (0,1), (1/40,1/8), (1/8,0), (9/40,1/8), (29/40,1), (7/8,29/40), (1,0).
G(y, z) = C'(f(y), f(z)) with C' = (1/12)[[11,1],[2,10]] is the dynamics on the cluster space {x1 = x2},
in the coordinates (y, z) = (x1 = x2, x3). The pair evolves in P, the singleton in Q.
"""
import numpy as np
from numba import njit

XS = np.array([0, 1/40, 1/8, 9/40, 29/40, 7/8, 1.0])
YS = np.array([1, 1/8, 0, 1/8, 1, 29/40, 0.0])
P, Q = (1/40, 9/40), (11/20, 7/8)

@njit(cache=True)
def f(x):
    for i in range(6):
        if x <= XS[i+1]: return YS[i] + (YS[i+1]-YS[i])*(x-XS[i])/(XS[i+1]-XS[i])
    return YS[-1]

@njit(cache=True)
def df(x):
    for i in range(6):
        if x <= XS[i+1]: return (YS[i+1]-YS[i])/(XS[i+1]-XS[i])
    return 0.0

@njit(cache=True)
def G(y, z):
    fy, fz = f(y), f(z)
    return 11/12*fy + 1/12*fz, fy/6 + 5/6*fz

@njit(cache=True)
def F(a, b, c):
    """One step of the full GCM with N = 3, eps = 1/4."""
    fa, fb, fc = f(a), f(b), f(c); m = (fa + fb + fc)/3
    return 0.75*fa + 0.25*m, 0.75*fb + 0.25*m, 0.75*fc + 0.25*m
