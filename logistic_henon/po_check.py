"""
Complements po_fast.py (Appendix B): (1) a period-6 saddle with positive transverse exponent exists at eps = 0.30215,
and its distance to long orbits on the attractor is computed; (2) census of
periodic orbits on the attractor at eps = 0.3029. Needs numpy, numba and scipy; uses about 1 GB of memory.
"""
import numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from scipy.spatial import cKDTree
from gcm_logistic import orbit
from po_fast import newton, gp, analyse
# 1) is the transversally unstable period-6 saddle at eps=0.30215 on the attractor?
eps = 0.30215
Y, Z = orbit(eps, 0.3, 0.8, 100000, 4_000_000)
rng = np.random.default_rng(3)
best = None
for k in rng.choice(len(Y), 3000, replace=False):
    y, z, ok = newton(eps, Y[k], Z[k], 6)
    if ok and abs(y-z) > 1e-8:
        tq = gp(eps, y, z, 6)[-1]
        if tq > 0: best = (y, z, tq); break
print("period-6 saddle with positive transverse exponent:", best)
y0, z0, _ = best
pts = [(y0, z0)]; a, b, c, d = 1-eps/3, eps/3, 2*eps/3, 1-2*eps/3
for _ in range(5):
    yy, zz = pts[-1]; fy, fz = 4*yy*(1-yy), 4*zz*(1-zz); pts.append((a*fy+b*fz, c*fy+d*fz))
pts = np.array(pts)
for n in (4_000_000, 40_000_000):
    Yb, Zb = orbit(eps, 0.3, 0.8, 100000, n)
    dmin = np.full(6, np.inf)
    for s in range(0, n, 4_000_000):
        tr = cKDTree(np.column_stack([Yb[s:s+4_000_000], Zb[s:s+4_000_000]]))
        dmin = np.minimum(dmin, tr.query(pts)[0])
    print(f"   orbit length {n:.0e}: max over the 6 points of the distance to the attractor sample = {dmin.max():.2e}")
sys.stdout.flush()
# 2) eps = 0.3029
on = analyse(0.3029, range(6, 37, 6), 800)
sad = [o for o in on if o[1] > 1 > o[2]]
print(f"eps=0.3029: {len(on)} periodic orbits on the attractor; saddles {len(sad)}, sinks {sum(o[1] < 1 for o in on)}; periods {sorted(set(o[0] for o in on))}")
print(f"   transverse exponents: min {min(o[3] for o in sad):+.3f}, max {max(o[3] for o in sad):+.3f}; unstable {sum(o[3] > 0 for o in sad)}")
for o in sorted(sad, key=lambda o: -o[3])[:4]:
    print(f"      period {o[0]:2d}: multipliers {o[1]:.3g}, {o[2]:.2g}; transverse exponent {o[3]:+.4f}; dist {o[4]:.1e}")
