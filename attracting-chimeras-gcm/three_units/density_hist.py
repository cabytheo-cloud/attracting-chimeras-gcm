"""
Empirical invariant density of G on the attractor, for Figure 1(c) and density_check.py.

4e6 orbits started uniformly in R, 100 transient steps, then 250 recorded steps (1e9 points), binned on square
cells of side 1e-4. Saved to density_hist.npz. Needs numpy and numba; a few minutes (--quick: 1e8 points).
"""
import sys, time
import numpy as np
from numba import njit
from common import G, P, Q

y0, z0, h = 0.064, 0.653, 1.0e-4
NY, NZ = 820, 1930
@njit(cache=True)
def run(Y, Z, T0, T, H):
    for k in range(Y.size):
        y, z = Y[k], Z[k]
        for _ in range(T0): y, z = G(y, z)
        for _ in range(T):
            y, z = G(y, z); H[int((y-y0)/h), int((z-z0)/h)] += 1
rng = np.random.default_rng(11); H = np.zeros((NY, NZ), np.int64); t0 = time.time()
for _ in range(4):
    n = 100_000 if "--quick" in sys.argv else 1_000_000
    run(rng.uniform(*P, n), rng.uniform(*Q, n), 100, 250, H)
D = H/H.sum()/h**2
np.savez_compressed("density_hist.npz", D=D, y0=y0, z0=z0, h=h)
print(f"{H.sum():.2e} points in {time.time()-t0:.0f}s; occupied area {np.count_nonzero(H)*h*h:.5f}; mean density on the support {D[D > 0].mean():.1f}; "
      f"99.5% quantile {np.quantile(D[D > 0], 0.995):.1f}; max {D.max():.1f}; median count per occupied cell {np.median(H[H > 0]):.0f}")
