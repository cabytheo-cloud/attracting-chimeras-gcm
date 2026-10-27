"""
Long orbits fill the exact attractor (Section 5). Needs exact_levels.pkl, produced by exact_attractor.py.

64 orbits of 3e6 steps are recorded on a grid of cells of side 5e-4. Reports the area occupied by each orbit,
the fraction of the grid cells whose center lies in A_red = G^6(R) that are visited, and the number of visited
cells whose center lies outside A_red (cells crossed by the boundary). Needs numpy, numba and matplotlib (for
point-in-polygon tests). About a minute; --quick uses shorter orbits.
"""
import sys, pickle
import numpy as np
from numba import njit, prange
from matplotlib.path import Path
from common import G, P, Q

T = 300_000 if "--quick" in sys.argv else 3_000_000
A = pickle.load(open("exact_levels.pkl", "rb"))[6]          # G^6(R) as a list of convex polygons
ys = [float(p[0]) for poly in A for p in poly]; zs = [float(p[1]) for poly in A for p in poly]
print("exact A_red: y in [%.5f, %.5f], z in [%.5f, %.5f]" % (min(ys), max(ys), min(zs), max(zs)))

@njit(parallel=True, cache=True)
def occ(Y0, Z0, T, y0, z0, h, NY, NZ):
    out = np.zeros((Y0.size, NY, NZ), np.uint8)
    for k in prange(Y0.size):
        y, z = Y0[k], Z0[k]
        for _ in range(1000): y, z = G(y, z)
        for _ in range(T):
            y, z = G(y, z)
            out[k, int((y-y0)/h), int((z-z0)/h)] = 1
    return out

h = 5e-4; y0, z0 = 0.0, 0.5; NY, NZ = 500, 1000
rng = np.random.default_rng(1); n = 64
O = occ(rng.uniform(*P, n), rng.uniform(*Q, n), T, y0, z0, h, NY, NZ)
per = O.reshape(n, -1).sum(1)*h*h; U = O.max(0)
print("occupied area per orbit of %.0e steps: min %.5f, max %.5f; union %.5f; exact area of A_red %.6f"
      % (T, per.min(), per.max(), U.sum()*h*h, 25751997557087/2614053856870400))
paths = [Path(np.array([[float(p[0]), float(p[1])] for p in poly])) for poly in A]
cy, cz = np.meshgrid(y0 + (np.arange(NY)+0.5)*h, z0 + (np.arange(NZ)+0.5)*h, indexing="ij")
pts = np.c_[cy.ravel(), cz.ravel()]
inA = np.zeros(len(pts), bool)
for pa in paths: inA |= pa.contains_points(pts)
inA = inA.reshape(NY, NZ)
print("cells with center in A_red: %d (area %.5f); fraction of them visited: %.4f; visited cells with center outside A_red: %d"
      % (inA.sum(), inA.sum()*h*h, U[inA].mean(), (U.astype(bool) & ~inA).sum()))
