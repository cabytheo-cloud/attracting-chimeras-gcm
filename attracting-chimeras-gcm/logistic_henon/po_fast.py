"""
Periodic orbits of the reduced map lying on the attractor (Appendix B): Newton search from points of a long orbit,
for eps = 0.30215 and 0.3029. Reports their in-plane multipliers (saddle, sink or source) and their transverse
exponent (1/q) sum log((1-eps)|f'(y_t)|). Needs numpy, numba and scipy; several minutes.
"""
import numpy as np, warnings, sys
from numba import njit
from scipy.spatial import cKDTree
from gcm_logistic import orbit
warnings.filterwarnings("ignore")

@njit(cache=True)
def gp(eps, y, z, p):
    a, b, c, d = 1-eps/3, eps/3, 2*eps/3, 1-2*eps/3
    j11, j12, j21, j22 = 1.0, 0.0, 0.0, 1.0; tl = 0.0
    for _ in range(p):
        gy, gz = 4-8*y, 4-8*z
        m11, m12, m21, m22 = a*gy, b*gz, c*gy, d*gz
        j11, j12, j21, j22 = m11*j11+m12*j21, m11*j12+m12*j22, m21*j11+m22*j21, m21*j12+m22*j22
        tl += np.log(abs((1-eps)*gy))
        fy, fz = 4*y*(1-y), 4*z*(1-z); y, z = a*fy+b*fz, c*fy+d*fz
    return y, z, j11, j12, j21, j22, tl/p

@njit(cache=True)
def newton(eps, y, z, p):
    for _ in range(60):
        yp, zp, j11, j12, j21, j22, _t = gp(eps, y, z, p)
        F1, F2 = yp-y, zp-z
        a11, a12, a21, a22 = j11-1, j12, j21, j22-1
        det = a11*a22-a12*a21
        if not np.isfinite(det) or det == 0: return -1.0, -1.0, False
        s1 = (-F1*a22+F2*a12)/det; s2 = (-a11*F2+a21*F1)/det
        y += s1; z += s2
        if not (0 <= y <= 1 and 0 <= z <= 1): return -1.0, -1.0, False
        if abs(s1)+abs(s2) < 1e-14: break
    yp, zp, _a, _b, _c, _d, _t = gp(eps, y, z, p)
    return y, z, abs(yp-y)+abs(zp-z) < 1e-10

def analyse(eps, periods, n_starts, n_attr=4_000_000, tol=2e-3, seed=3):
    Y, Z = orbit(eps, 0.3, 0.8, 100000, n_attr)
    tree = cKDTree(np.column_stack([Y, Z]))
    rng = np.random.default_rng(seed); seen = set(); out = []
    for p in periods:
        for k in rng.choice(len(Y), n_starts, replace=False):
            y, z, ok = newton(eps, Y[k], Z[k], p)
            if not ok or abs(y-z) < 1e-8: continue
            q = next(q for q in range(1, p+1) if p % q == 0 and np.hypot(*(np.array(gp(eps, y, z, q)[:2])-[y, z])) < 1e-9)
            pts = [(y, z)]
            for _ in range(q-1):
                yy, zz = pts[-1]; a, b, c, d = 1-eps/3, eps/3, 2*eps/3, 1-2*eps/3
                fy, fz = 4*yy*(1-yy), 4*zz*(1-zz); pts.append((a*fy+b*fz, c*fy+d*fz))
            key = (q, round(min(pts)[0], 7))
            if key in seen: continue
            seen.add(key)
            dist = tree.query(np.array(pts))[0].max()
            if dist > tol: continue
            _y, _z, j11, j12, j21, j22, tq = gp(eps, y, z, q)
            mult = np.sort(np.abs(np.linalg.eigvals(np.array([[j11, j12], [j21, j22]]))))[::-1]
            out.append((q, mult[0], mult[1], tq, dist))
    return sorted(out)

if __name__ == "__main__":
    for eps, periods in ((0.30215, range(1, 31)), (0.3029, range(6, 49, 6))):
        on = analyse(eps, periods, 1500)
        sad = [o for o in on if o[1] > 1 > o[2]]
        print(f"eps={eps}: {len(on)} periodic orbits on the attractor (within 2e-3); saddles {len(sad)}, "
              f"sinks {sum(o[1] < 1 for o in on)}, sources {sum(o[2] > 1 for o in on)}; periods {sorted(set(o[0] for o in on))}")
        tp = [o for o in sad if o[3] > 0]
        print(f"   transverse exponents of the saddles: min {min(o[3] for o in sad):+.3f}, max {max(o[3] for o in sad):+.3f}; "
              f"transversally unstable: {len(tp)} of {len(sad)}")
        for o in sorted(sad, key=lambda o: -o[3])[:4]:
            print(f"      period {o[0]:2d}: multipliers {o[1]:.3g}, {o[2]:.2g}; transverse exponent {o[3]:+.4f}; dist {o[4]:.1e}")
        sys.stdout.flush()
