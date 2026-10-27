"""Search for 6-piece Markov maps (steep down | -5/4 | +5/4 | steep up | down | steep down) whose block
valley p sits close to the block's mean image, so that the chimera attractor is not split into two bands.
Nodes 0 < a < p < k < c < d < 1 ; g(0)=1, g(p)=0, g(c)=1, g(1)=0, g(a), g(k), g(d) in the node set."""
import numpy as np, itertools, pickle
from fractions import Fraction as F
Cp = np.array([[11/12, 1/12], [1/6, 5/6]])
def sig(a, b): return np.linalg.svd(Cp @ np.diag([a, b]), compute_uv=False).min()
def nice(lo, hi, dens):
    return sorted({F(n, d) for d in dens for n in range(1, d) if lo <= F(n, d) <= hi})
def primitive(XS, YS):
    n = len(XS) - 1; A = np.zeros((n, n), int)
    for i in range(n):
        lo, hi = sorted([YS[i], YS[i+1]])
        for j in range(n):
            if lo <= XS[j] and XS[j+1] <= hi: A[i, j] = 1
    M = A.copy()
    for _ in range(12):
        if (M > 0).all(): return True
        M = np.minimum(M @ A, 1)
    return False
G = 14
res = []
ps = nice(F(17, 200), F(3, 20), [24, 30, 36, 40, 48, 60, 72, 80, 96, 120])
cs = nice(F(3, 5), F(4, 5), [24, 30, 36, 40, 48, 60, 72])
for ip, p in enumerate(ps):
    print("p", ip, len(ps), p, len(res), flush=True)
    for aopt, kopt in itertools.product(range(2), range(3)):
        a, ga = (p/5, p) if aopt == 0 else (5*p/9, 5*p/9)
        if kopt == 0: k, gk = 9*p/5, p
        elif kopt == 1: k, gk = p + 4*a/5, a
        else: k, gk = 5*p, 5*p
        for c in cs:
            if c <= k + F(1, 20): continue
            s4 = (1 - gk)/(c - k)
            if s4 < F(7, 5): continue
            for gd in {a, p, k, c, ga, gk}:
                for dd in [24, 30, 36, 40, 48, 60, 72]:
                    for n in range(1, dd):
                        d = F(n, dd)
                        if not (c < d < 1): continue
                        s5 = (1 - gd)/(d - c); s6 = gd/(1 - d)
                        if s5 < F(7, 5) or s5 > 5 or s6 < F(6, 5): continue
                        sg = min(sig(1.25, float(s4)), sig(1.25, float(s5)))
                        if sg < 1.015: continue
                        pf, af, kf, cf, df = map(float, (p, a, k, c, d))
                        Pa = np.linspace(af, pf, G, endpoint=False)[:, None, None, None]
                        Pb = np.linspace(pf, kf, G + 1)[1:][None, :, None, None]
                        l = np.linspace(kf, cf, G, endpoint=False)[None, None, :, None]
                        e = np.linspace(cf, df, G + 1)[1:][None, None, None, :]
                        w = 1.25*np.maximum(pf - Pa, Pb - pf)
                        m = np.minimum(1 - float(s4)*(cf - l), 1 - float(s5)*(e - cf))
                        M = np.minimum.reduce([np.broadcast_to(x, (G, G, G, G)) for x in
                              (m/12 - Pa, Pb - 11/12*w - 1/12, 5/6*m - l, e - 5/6 - w/6, m - 0)])
                        idx = np.unravel_index(M.argmax(), M.shape); mg = M[idx]
                        if mg <= 0.004: continue
                        XS = [F(0), a, p, k, c, d, F(1)]; YS = [F(1), ga, F(0), gk, F(1), gd, F(0)]
                        if not primitive(XS, YS): continue
                        Pv = (Pa.ravel()[idx[0]], Pb.ravel()[idx[1]]); Qv = (l.ravel()[idx[2]], e.ravel()[idx[3]])
                        mm = m[0, 0, idx[2], idx[3]] if m.shape[2] > 1 else None
                        mm = min(1 - float(s4)*(cf - Qv[0]), 1 - float(s5)*(Qv[1] - cf))
                        ratio = ((1 - mm)/12)/(pf - (1 + mm)/24)
                        res.append(dict(XS=XS, YS=YS, P=Pv, Q=Qv, margin=mg, sigma=sg, ratio=ratio,
                                        slopes=[float((YS[i+1]-YS[i])/(XS[i+1]-XS[i])) for i in range(6)]))
print(len(res), "candidates")
pickle.dump(res, open("cands.pkl", "wb"))
res.sort(key=lambda r: -min(r["margin"], (r["sigma"] - 1)/3))
for r in res[:25]:
    print(f"margin {r['margin']:.4f} sigma {r['sigma']:.4f} ratio {r['ratio']:.2f} XS {[str(x) for x in r['XS']]} YS {[str(y) for y in r['YS']]} slopes {np.round(r['slopes'],3).tolist()}")
