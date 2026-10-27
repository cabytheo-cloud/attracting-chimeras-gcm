"""
Basins of the four-piece example (Section 6.2): fate of 2x10^6 initial conditions drawn uniformly in [0,1]^3
after 4000 steps (paper: the chimeras attract only about a quarter of them). Needs numpy and numba; a few minutes
(--quick: 2x10^5 initial conditions).
"""
import sys
import numpy as np, time
from numba import njit, prange
@njit(cache=True)
def f(x):
    if x <= 0.125: return 1.25*x
    if x <= 0.25: return 0.3125 - 1.25*x
    if x <= 0.75: return 2*x - 0.5
    return 2.5 - 2*x
@njit(parallel=True, cache=True)
def classify(X, T):
    n = X.shape[0]; out = np.empty(n, np.int64); tin = np.empty(n, np.int64)
    for k in prange(n):
        a, b, c = X[k, 0], X[k, 1], X[k, 2]; tk = -1
        for t in range(T):
            fa, fb, fc = f(a), f(b), f(c); m = (fa + fb + fc)/3
            a, b, c = 0.75*fa + 0.25*m, 0.75*fb + 0.25*m, 0.75*fc + 0.25*m
            if tk < 0 and (abs(a-b) < 1e-12 or abs(a-c) < 1e-12 or abs(b-c) < 1e-12): tk = t
        d = np.array([abs(b-c), abs(a-c), abs(a-b)])   # pair opposite to unit 0,1,2
        small = d < 1e-10
        if small.sum() == 3: out[k] = 3                  # full synchrony
        elif small.sum() == 1:
            j = np.argmin(d); out[k] = j                 # chimera; j = index of the unsynchronized unit
        else: out[k] = 4                                 # no pair synchronized
        tin[k] = tk
    return out, tin
rng = np.random.default_rng(0); n = 200_000 if '--quick' in sys.argv else 2_000_000
X = rng.uniform(0, 1, (n, 3))
t0 = time.time(); o, tin = classify(X, 4000); print(f"{time.time()-t0:.0f}s")
names = {0: "chimera, unit 1 apart", 1: "chimera, unit 2 apart", 2: "chimera, unit 3 apart", 3: "full synchrony", 4: "no synchronized pair"}
for k in range(5):
    p = np.mean(o == k); print(f"  {names[k]:24s} {p:.5f}  (+- {1.96*np.sqrt(p*(1-p)/n):.5f})")
ch = o <= 2
print(f"  all chimeras: {ch.mean():.5f};  median time to first near-merge {int(np.median(tin[tin>=0]))}, 99.9% quantile {int(np.quantile(tin[tin>=0],0.999))}")
