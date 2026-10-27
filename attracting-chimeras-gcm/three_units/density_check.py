"""
Boundedness of the invariant density (Section 5), without binning. Needs density_hist.npz (density_hist.py).

The image of the normalized Lebesgue measure on R under G^n has density
  h_n(p) = (1/area(R)) * sum over backward branches p = G(q_1), q_1 = G(q_2), ..., q_{n-1} = G(q_n) staying in R
           of prod_k 1/|det DG(q_k)|,
where |det DG| = (3/4)(5/4)(7/4) = 105/64 on the left lap of f|Q and (3/4)(5/4)(11/6) = 55/32 on the right lap.
h_n is evaluated along short segments through the three cells where the empirical density is largest;
its maximum stays bounded as n grows. Needs numpy; a few minutes.
"""
import numpy as np
Ci = np.array([[10/9, -1/9], [-2/9, 11/9]])           # inverse of C' = (1/12)[[11,1],[2,10]]
R_area = (9/40 - 1/40)*(7/8 - 11/20)
def h(p, n):
    y = np.array([p[0]]); z = np.array([p[1]]); w = np.array([1.0])
    for k in range(n):
        u = Ci[0, 0]*y + Ci[0, 1]*z; v = Ci[1, 0]*y + Ci[1, 1]*z
        ok = (u >= 0) & (u <= 1/8) & (v >= 111/160) & (v <= 1)
        u, v, w = u[ok], v[ok], w[ok]
        yl, yr = 1/8 - 0.8*u, 1/8 + 0.8*u                       # both laps of f|P map onto [0, 1/8]
        zl = 29/40 - 4*(1 - v)/7; zr = 29/40 + 6*(1 - v)/11       # left lap onto [111/160, 1], right lap onto [29/40, 1]
        okr = v >= 29/40
        y = np.concatenate([yl, yr, yl[okr], yr[okr]]); z = np.concatenate([zl, zl, zr[okr], zr[okr]])
        w = np.concatenate([w*64/105, w*64/105, w[okr]*32/55, w[okr]*32/55])
        if y.size > 3e7: raise RuntimeError("too many branches")
    return w.sum()/R_area
dh = np.load("density_hist.npz"); D, y0, z0, hh = dh["D"], float(dh["y0"]), float(dh["z0"]), float(dh["h"])
idx = np.argsort(D.ravel())[::-1]; picks = []
for i in idx:
    iy, iz = np.unravel_index(i, D.shape)
    if all(abs(iy - a) + abs(iz - b) > 150 for a, b in picks): picks.append((iy, iz))
    if len(picks) == 3: break
print("mean density 1/area(A) =", 2614053856870400/25751997557087)
for iy, iz in picks:
    yc, zc = y0 + (iy + 0.5)*hh, z0 + (iz + 0.5)*hh
    pts = [(y, zc) for y in np.linspace(yc - 0.001, yc + 0.001, 40)]
    print(f"segment z={zc:.4f}, y in [{yc-0.001:.4f},{yc+0.001:.4f}] (empirical density there {D[iy, iz]:.0f})")
    for n in (12, 16, 20, 21, 22):
        print(f"   n={n:2d}: max h_n = {max(h(p, n) for p in pts):7.1f}", flush=True)
