"""
The reduced map with 60 significant digits (mpmath), compared with double precision (Appendix B). If the chaotic
attractor were an artefact of rounding, the high-precision orbit would settle on a sink. Needs numpy, numba and
mpmath; several minutes.
"""
import mpmath as mp, numpy as np, time
from gcm_logistic import orbit
mp.mp.dps = 60
def lam1_hp(eps, y, z, n_trans, n):
    eps = mp.mpf(eps); a, b, c, d = 1-eps/3, eps/3, 2*eps/3, 1-2*eps/3
    y, z = mp.mpf(y), mp.mpf(z)
    for _ in range(n_trans):
        fy, fz = 4*y*(1-y), 4*z*(1-z); y, z = a*fy+b*fz, c*fy+d*fz
    v1, v2 = mp.mpf(1), mp.mpf(0); s = mp.mpf(0); Y = []; Z = []
    for t in range(n):
        gy, gz = 4-8*y, 4-8*z
        w1 = a*gy*v1+b*gz*v2; w2 = c*gy*v1+d*gz*v2
        r = mp.sqrt(w1*w1+w2*w2); v1, v2 = w1/r, w2/r; s += mp.log(r)
        fy, fz = 4*y*(1-y), 4*z*(1-z); y, z = a*fy+b*fz, c*fy+d*fz
        if t % 10 == 0: Y.append(float(y)); Z.append(float(z))
    return float(s/n), np.array(Y), np.array(Z)
t0 = time.time()
for eps in ("0.30215", "0.3029"):
    l, Y, Z = lam1_hp(eps, "0.3", "0.8", 20000, 150000)
    Yd, Zd = orbit(float(eps), 0.3, 0.8, 20000, 150000)
    # compare the two point clouds (attractor shape) via 2D histograms
    H1, xe, ye = np.histogram2d(Y, Z, bins=60, range=[[0, 1], [0, 1]]); H2, _, _ = np.histogram2d(Yd[::10], Zd[::10], bins=[xe, ye])
    overlap = np.minimum(H1/H1.sum(), H2/H2.sum()).sum()
    print(f"eps={eps}: lam1 with 60 digits over 1.5e5 steps = {l:+.4f};  histogram overlap with double-precision run = {overlap:.3f}   ({time.time()-t0:.0f}s)", flush=True)
