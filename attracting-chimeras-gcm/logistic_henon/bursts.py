"""
The full system (N = 3) from 300 random initial conditions in [0,1]^3, for eps = 0.30215, 0.3025, 0.3029 (Appendix B).
Once two units are within 1e-10, their distance is followed exactly in logarithmic scale, using
f(y+h) - f(y-h) = 2 f'(y) h for f(x) = 4x(1-x), so that rounding cannot merge the two units. Over 2x10^6 further
steps, the log-distance decreases at the rate given by the TLE, with no desynchronization burst.
Needs numpy and numba; about 10 minutes.
"""
import numpy as np, time
from numba import njit
@njit(cache=True)
def one(eps, x1, x2, x3, T):
    a = 1 - eps; x = np.array([x1, x2, x3]); t = 0; pair = -1
    while t < 200000:
        f = 4*x*(1-x); m = f.mean(); x = a*f + eps*m; t += 1
        if abs(x[0]-x[1]) < 1e-10: pair = 2; break
        if abs(x[0]-x[2]) < 1e-10: pair = 1; break
        if abs(x[1]-x[2]) < 1e-10: pair = 0; break
    if pair < 0: return t, 0.0, 0.0, 0.0
    i, j = (0, 1) if pair == 2 else ((0, 2) if pair == 1 else (1, 2))
    y = 0.5*(x[i]+x[j]); z = x[pair]; L = np.log(abs(x[i]-x[j])); Lmin = L; exc = 0.0; Lmax = L
    for s in range(T):
        dl2 = np.exp(2*L) if L > -350 else 0.0
        fy, fz = 4*y*(1-y), 4*z*(1-z); m = (2*fy - 2*dl2 + fz)/3
        L += np.log(abs(a*(4 - 8*y))); y, z = a*(fy - dl2) + eps*m, a*fz + eps*m
        if L < Lmin: Lmin = L
        if L - Lmin > exc: exc = L - Lmin
        if L > Lmax: Lmax = L
    return t, L/T, exc, Lmax
rng = np.random.default_rng(5)
for eps in (0.30215, 0.3025, 0.3029):
    t0 = time.time(); R = np.array([one(eps, *rng.uniform(0, 1, 3), 2_000_000) for _ in range(300)])
    ok = R[:, 1] != 0
    print(f"eps={eps}: {ok.sum()}/300 ICs reach |x_i-x_j|<1e-10 (median time {int(np.median(R[ok,0]))}); then over 2e6 steps: "
          f"rate {R[ok,1].mean():+.4f} [{R[ok,1].min():+.4f}, {R[ok,1].max():+.4f}], largest rise of log|x_i-x_j| above its running min: {R[ok,2].max():.1f} "
          f"(e-folds), max log|x_i-x_j| after switch {R[ok,3].max():.1f}   ({time.time()-t0:.0f}s)", flush=True)
