"""
Globally coupled logistic maps f(x) = 4x(1-x), N = 3 (Appendix B). Shared functions for the other scripts.

Reduced map on the cluster space {x1 = x2}, in the coordinates (y, z) = (x1 = x2, x3):
G(y, z) = C_eps (f(y), f(z)), C_eps = [[1-eps/3, eps/3], [2eps/3, 1-2eps/3]].
The TLE of {x1 = x2} along an orbit is log(1-eps) + <log|f'(y)|>. Lyapunov exponents of G by the QR method.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def f(x):
    return 4.0*x*(1.0 - x)

@njit(cache=True)
def df(x):
    return 4.0 - 8.0*x

@njit(cache=True)
def run(eps, y, z, n_trans, n):
    """Returns tle, lam1, lam2, final y, z, min distance |y - z| seen."""
    a, b, c, d = 1 - eps/3, eps/3, 2*eps/3, 1 - 2*eps/3
    for _ in range(n_trans):
        fy, fz = f(y), f(z)
        y, z = a*fy + b*fz, c*fy + d*fz
    # tangent frame
    v11, v21, v12, v22 = 1.0, 0.0, 0.0, 1.0
    s1 = 0.0; s2 = 0.0; st = 0.0; dmin = 1.0
    for _ in range(n):
        gy, gz = df(y), df(z)
        st += np.log(abs(gy))
        # J = C diag(gy, gz)
        j11, j12, j21, j22 = a*gy, b*gz, c*gy, d*gz
        w11 = j11*v11 + j12*v21; w21 = j21*v11 + j22*v21
        w12 = j11*v12 + j12*v22; w22 = j21*v12 + j22*v22
        # Gram-Schmidt
        r1 = np.sqrt(w11*w11 + w21*w21)
        v11, v21 = w11/r1, w21/r1
        p = v11*w12 + v21*w22
        w12 -= p*v11; w22 -= p*v21
        r2 = np.sqrt(w12*w12 + w22*w22)
        v12, v22 = w12/r2, w22/r2
        s1 += np.log(r1); s2 += np.log(r2)
        fy, fz = f(y), f(z)
        y, z = a*fy + b*fz, c*fy + d*fz
        dd = abs(y - z)
        if dd < dmin: dmin = dd
    return np.log(1 - eps) + st/n, s1/n, s2/n, y, z, dmin

@njit(cache=True)
def orbit(eps, y, z, n_trans, n):
    a, b, c, d = 1 - eps/3, eps/3, 2*eps/3, 1 - 2*eps/3
    for _ in range(n_trans):
        fy, fz = f(y), f(z)
        y, z = a*fy + b*fz, c*fy + d*fz
    Y = np.empty(n); Z = np.empty(n)
    for t in range(n):
        fy, fz = f(y), f(z)
        y, z = a*fy + b*fz, c*fy + d*fz
        Y[t] = y; Z[t] = z
    return Y, Z
