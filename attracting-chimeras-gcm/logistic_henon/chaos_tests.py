"""
Robustness of the positive Lyapunov exponent of the Henon-type attractors (Appendix B):
  1) the largest exponent with uniform random perturbations of size eta in {0, 1e-14, 1e-10, 1e-6, 1e-4} at each step;
  2) the largest exponent over 100 consecutive blocks of 10^7 steps along one orbit of 10^9 steps (detects a late
     collapse onto a periodic sink). Needs numpy and numba; about 10 minutes.
"""
import numpy as np, time
from numba import njit
from gcm_logistic import run, orbit, f, df

@njit(cache=True)
def run_noise(eps, y, z, n_trans, n, eta, seed):
    """Same as run (lam1 only) but with uniform noise of amplitude eta added each step."""
    np.random.seed(seed)
    a, b, c, d = 1 - eps/3, eps/3, 2*eps/3, 1 - 2*eps/3
    for _ in range(n_trans):
        fy, fz = f(y), f(z)
        y = a*fy + b*fz + eta*(2*np.random.random() - 1); z = c*fy + d*fz + eta*(2*np.random.random() - 1)
    v1, v2 = 1.0, 0.0; s1 = 0.0
    for _ in range(n):
        gy, gz = df(y), df(z)
        w1 = a*gy*v1 + b*gz*v2; w2 = c*gy*v1 + d*gz*v2
        r = np.sqrt(w1*w1 + w2*w2); v1, v2 = w1/r, w2/r; s1 += np.log(r)
        fy, fz = f(y), f(z)
        y = a*fy + b*fz + eta*(2*np.random.random() - 1); z = c*fy + d*fz + eta*(2*np.random.random() - 1)
    return s1/n

@njit(cache=True)
def windows(eps, y, z, n_win, win):
    """lam1 over consecutive windows of a single very long orbit (detects late collapse onto a sink)."""
    a, b, c, d = 1 - eps/3, eps/3, 2*eps/3, 1 - 2*eps/3
    out = np.empty(n_win); v1, v2 = 1.0, 0.0
    for k in range(n_win):
        s = 0.0
        for _ in range(win):
            gy, gz = df(y), df(z)
            w1 = a*gy*v1 + b*gz*v2; w2 = c*gy*v1 + d*gz*v2
            r = np.sqrt(w1*w1 + w2*w2); v1, v2 = w1/r, w2/r; s += np.log(r)
            fy, fz = f(y), f(z); y, z = a*fy + b*fz, c*fy + d*fz
        out[k] = s/win
    return out

if __name__ == "__main__":
    EPS = [0.30215, 0.3025, 0.3029, 0.3031]
    print("1) robustness to noise (lam1), 2e6 steps")
    for eps in EPS:
        vals = [run_noise(eps, 0.3, 0.8, 200000, 2_000_000, eta, 5) for eta in (0.0, 1e-14, 1e-10, 1e-6, 1e-4)]
        print(f"   eps={eps}: " + "  ".join(f"eta={eta:.0e}: {v:+.4f}" for eta, v in zip((0, 1e-14, 1e-10, 1e-6, 1e-4), vals)))
    print("2) single orbit of 1e9 steps, lam1 per window of 1e7 steps: min / max / last")
    t0 = time.time()
    for eps in EPS:
        w = windows(eps, 0.3, 0.8, 100, 10_000_000)
        print(f"   eps={eps}: min {w.min():+.4f}  max {w.max():+.4f}  last {w[-1]:+.4f}   ({time.time()-t0:.0f}s)", flush=True)
