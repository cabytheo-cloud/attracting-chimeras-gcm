"""
Numerical illustration of the main example (Section 5).

  1. Lyapunov exponents of G on the chimera and Birkhoff average of the TLE along three long orbits
     (paper: about 0.41 and 0.12; TLE log(15/16)).
  2. Fate of 10^6 initial conditions drawn uniformly in [0,1]^3: share of each of the three chimeras
     (paper: a third each; full synchrony is never observed).
  3. Time needed to enter one of the three trapping boxes (paper: median 71 steps).
Needs numpy and numba. Full run: a few minutes on a laptop. Use --quick for a fast, smaller run.
"""
import sys, time
import numpy as np
from numba import njit, prange
from common import f, df, F, P, Q

QUICK = "--quick" in sys.argv
N_LYAP, N_IC, N_CAPT = (2_000_000, 50_000, 20_000) if QUICK else (20_000_000, 1_000_000, 200_000)

@njit(cache=True)
def exponents(y, z, n):
    """TLE log((3/4)|f'(y)|) and the two Lyapunov exponents of G (QR method) along one orbit."""
    a, b, c, d = 11/12, 1/12, 1/6, 5/6
    for _ in range(1000):
        fy, fz = f(y), f(z); y, z = a*fy + b*fz, c*fy + d*fz
    v11, v21, v12, v22 = 1.0, 0.0, 0.0, 1.0; s1 = 0.0; s2 = 0.0; st = 0.0
    for _ in range(n):
        gy, gz = df(y), df(z); st += np.log(0.75*abs(gy))
        j11, j12, j21, j22 = a*gy, b*gz, c*gy, d*gz
        w11 = j11*v11 + j12*v21; w21 = j21*v11 + j22*v21; w12 = j11*v12 + j12*v22; w22 = j21*v12 + j22*v22
        r1 = np.sqrt(w11*w11 + w21*w21); v11, v21 = w11/r1, w21/r1
        p = v11*w12 + v21*w22; w12 -= p*v11; w22 -= p*v21
        r2 = np.sqrt(w12*w12 + w22*w22); v12, v22 = w12/r2, w22/r2
        s1 += np.log(r1); s2 += np.log(r2)
        fy, fz = f(y), f(z); y, z = a*fy + b*fz, c*fy + d*fz
    return st/n, s1/n, s2/n

@njit(parallel=True, cache=True)
def classify(X, T):
    """0: full synchrony; 1, 2, 3: chimera with the pair {2,3}, {1,3}, {1,2}; 4: no synchronized pair."""
    n = X.shape[0]; out = np.empty(n, np.int64)
    for k in prange(n):
        a, b, c = X[k, 0], X[k, 1], X[k, 2]
        for t in range(T): a, b, c = F(a, b, c)
        d12, d13, d23 = abs(a-b), abs(a-c), abs(b-c)
        if d12 < 1e-10 and d13 < 1e-10: out[k] = 0
        elif d23 < 1e-10: out[k] = 1
        elif d13 < 1e-10: out[k] = 2
        elif d12 < 1e-10: out[k] = 3
        else: out[k] = 4
    return out

@njit(parallel=True, cache=True)
def capture_time(X, T):
    """First time at which two units lie in int P and one in int Q (that is, the orbit enters a trapping box)."""
    n = X.shape[0]; out = np.full(n, -1, np.int64)
    for k in prange(n):
        a, b, c = X[k, 0], X[k, 1], X[k, 2]
        for t in range(T):
            a, b, c = F(a, b, c)
            inP = (P[0] < a < P[1]) + (P[0] < b < P[1]) + (P[0] < c < P[1])
            inQ = (Q[0] < a < Q[1]) + (Q[0] < b < Q[1]) + (Q[0] < c < Q[1])
            if inP == 2 and inQ == 1:
                out[k] = t + 1; break
    return out

if __name__ == "__main__":
    print(f"1. Exponents on the chimera ({N_LYAP:.0e} steps per orbit; log(15/16) = {np.log(15/16):.6f})")
    for y0, z0 in ((0.06, 0.8), (0.12, 0.7), (0.2, 0.85)):
        tle, l1, l2 = exponents(y0, z0, N_LYAP)
        print(f"   from ({y0}, {z0}): TLE {tle:.6f}, Lyapunov exponents of G {l1:.4f}, {l2:.4f}", flush=True)

    t0 = time.time(); X = np.random.default_rng(4).uniform(0, 1, (N_IC, 3)); o = classify(X, 6000)
    rest = np.flatnonzero(o == 4)
    if rest.size: o[rest] = classify(X[rest], 46000)       # orbits not yet settled after 6000 steps: run longer
    shares = [np.mean(o == k) for k in (1, 2, 3)]
    print(f"2. {N_IC} random initial conditions in [0,1]^3: chimeras with pair {{2,3}}, {{1,3}}, {{1,2}}: "
          + ", ".join(f"{x:.4f}" for x in shares) + f"; full synchrony: {np.sum(o == 0)}; no synchronized pair: {np.sum(o == 4)}"
          + f"  ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time(); X = np.random.default_rng(9).uniform(0, 1, (N_CAPT, 3)); tc = capture_time(X, 2_000_000); caught = tc >= 0
    print(f"3. {N_CAPT} random initial conditions: entered a trapping box within 2e6 steps: {caught.mean():.6f}  ({time.time()-t0:.0f}s)")
    if caught.any():
        q = np.quantile(tc[caught], [0.5, 0.9, 0.99, 0.999, 1.0]).astype(int).tolist()
        print(f"   capture time: median {q[0]}, quantiles 90/99/99.9%: {q[1]}/{q[2]}/{q[3]}, max {q[4]}")
