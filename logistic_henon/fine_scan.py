"""
Scan of eps in [0.3015, 0.304] (Appendix B): TLE and Lyapunov exponents of the reduced map from 3 random initial
conditions per value (10^6 transient + 2x10^6 recorded steps). The chaotic window is eps in [0.3021, 0.3032],
apart from a narrow periodic window near 0.3028. Saves fine_scan.npy. Needs numpy and numba; a few minutes.
"""
import numpy as np, time
from gcm_logistic import run
rng = np.random.default_rng(1)
t0 = time.time()
eps_list = np.round(np.arange(0.3015, 0.30401, 0.00002), 6)
rows = []
for eps in eps_list:
    for k in range(3):
        y, z = rng.uniform(0, 1, 2)
        tle, l1, l2, yf, zf, dmin = run(eps, y, z, 1_000_000, 2_000_000)
        rows.append((eps, k, tle, l1, l2))
rows = np.array(rows); np.save("fine_scan.npy", rows)
print(f"{time.time()-t0:.0f}s")
for eps in eps_list[::4]:
    r = rows[rows[:, 0] == eps]
    print(f"{eps:.5f}  TLE {r[:,2].min():+.4f}..{r[:,2].max():+.4f}   lam1 {r[:,3].min():+.4f}..{r[:,3].max():+.4f}   lam2 {r[:,4].mean():+.4f}")
