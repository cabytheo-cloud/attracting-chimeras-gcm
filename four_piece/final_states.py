"""
Where the orbits that do not converge to a chimera go, in the four-piece example (Section 6.2): 10^6 initial
conditions in [0,1]^3, 6000 steps. Classifies the final states (full synchrony, chimera, no synchronized pair)
and counts how many units sit in the small tent [0,1/4] (below 0.4) over the last 1000 steps
(paper: full synchrony occurs in the small tent). Needs numpy and numba (--quick: 10^5 initial conditions).
"""
import sys
import numpy as np
from numba import njit, prange
@njit(cache=True)
def f(x):
    if x <= 0.125: return 1.25*x
    if x <= 0.25: return 0.3125 - 1.25*x
    if x <= 0.75: return 2*x - 0.5
    return 2.5 - 2*x
@njit(parallel=True, cache=True)
def final(X, T):
    n = X.shape[0]; out = np.empty((n, 3)); lo = np.zeros(n, np.int64)
    for k in prange(n):
        a, b, c = X[k, 0], X[k, 1], X[k, 2]
        for t in range(T):
            fa, fb, fc = f(a), f(b), f(c); m = (fa+fb+fc)/3
            a, b, c = 0.75*fa+0.25*m, 0.75*fb+0.25*m, 0.75*fc+0.25*m
            if t >= T - 1000:      # count steps with a unit below 0.4 in the last 1000 steps
                lo[k] += (a < 0.4) + (b < 0.4) + (c < 0.4)
        out[k] = a, b, c
    return out, lo
rng = np.random.default_rng(1); n = 100_000 if '--quick' in sys.argv else 1_000_000
X = rng.uniform(0, 1, (n, 3)); Y, lo = final(X, 6000)
d = np.sort(np.abs(np.stack([Y[:,1]-Y[:,2], Y[:,0]-Y[:,2], Y[:,0]-Y[:,1]], 1)), 1)
full = d[:, 2] < 1e-10; chim = (d[:, 0] < 1e-10) & (d[:, 1] > 1e-6); none = d[:, 0] > 1e-10
nlow = lo / 1000.0        # average number of units below 0.4
print(f"full synchrony {full.mean():.4f}   chimeras {chim.mean():.4f}   no synchronized pair {none.mean():.4f}")
for name, sel in (("full synchrony", full), ("chimeras", chim), ("no pair", none)):
    h = np.round(nlow[sel]).astype(int); print(f"  {name}: average number of units in the low region: " + ", ".join(f"{k}: {np.mean(h==k):.3f}" for k in range(4)))
