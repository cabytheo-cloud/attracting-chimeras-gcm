"""
Birkhoff averages on the attractor (Section 5): 4000 orbits started uniformly in R, 300 transient steps, 500 recorded steps.

  - the TLE, average of log((3/4)|f'(y)|), equals log(15/16);
  - the average of Proposition 4.3, log(3/4) + log|f[y,z]| with f[y,z] = (f(y) - f(z))/(y - z), vanishes;
  - the area occupied by the orbits on a 300 x 300 grid is close to the exact area of A_red.
Needs numpy and numba; a few seconds.
"""
import numpy as np
from numba import njit, prange
from common import f, df, G, P, Q

@njit(parallel=True, cache=True)
def run(Y0, Z0, T0, T, nb, y0, z0, w, hgt):
    n = Y0.size; s1 = np.zeros(n); s2 = np.zeros(n); occ = np.zeros((n, nb, nb), np.uint8)
    for k in prange(n):
        y, z = Y0[k], Z0[k]
        for _ in range(T0): y, z = G(y, z)
        for _ in range(T):
            s1[k] += np.log(0.75*abs(df(y)))
            s2[k] += np.log(0.75) + np.log(abs((f(y) - f(z))/(y - z)))
            i = int((y-y0)/w*nb); j = int((z-z0)/hgt*nb)
            if 0 <= i < nb and 0 <= j < nb: occ[k, i, j] = 1
            y, z = G(y, z)
    return s1.sum()/(n*T), s2.sum()/(n*T), occ

rng = np.random.default_rng(3)
y0, z0, w, hgt, nb = 0.065, 0.654, 0.08, 0.192, 300
a, b, O = run(rng.uniform(*P, 4000), rng.uniform(*Q, 4000), 300, 500, nb, y0, z0, w, hgt)
print(f"average of log|(3/4) f'(y)| = {a:.6f}   (log 15/16 = {np.log(15/16):.6f})")
print(f"average of log(3/4) + log|f[y,z]| = {b:.6f}   (Proposition 4.3: 0)")
O = O.max(axis=0)
print(f"occupied area on a {nb}x{nb} grid: {O.sum()*w*hgt/nb**2:.5f}   (exact area of A_red: {25751997557087/2614053856870400:.6f})")
