"""
Exact verification of the main example (N = 3, eps = 1/4, six-piece local map f).

Checks, in exact rational arithmetic, every number used in the statements and proofs of
  Proposition 3.1  (Markov partition, transitions, invariant density of f)        Appendix A.2
  Theorem 3.2 (i)  (trapping region U = P x P x Q, margin 1/48)                    Appendix A.3
  Theorem 3.2 (iv) (invariant parallelogram S, Lemma A.1)                          Appendix A.6
  Proposition 3.3  (exact attractor: G^7(R) = G^6(R), computer-assisted)           Appendix A.7
  Proposition 3.4  (uniform expansion of G on the four rectangles)                 Appendix A.8
Only the Python standard library is needed. Run time: about 40 s (most of it for Proposition 3.3).
"""
import sys
from fractions import Fraction as F
import math
import exact_polygons as L

FAILS = []
def ok(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond: FAILS.append(msg)
s = lambda x: str(x)

# ---------------------------------------------------------------------------------------------
# The local map f: piecewise linear interpolation of the points (2) of the paper
XS = [F(0), F(1,40), F(1,8), F(9,40), F(29,40), F(7,8), F(1)]
YS = [F(1), F(1,8), F(0), F(1,8), F(1), F(29,40), F(0)]
SL = [(YS[i+1]-YS[i])/(XS[i+1]-XS[i]) for i in range(6)]
def f(x):
    for i in range(6):
        if XS[i] <= x <= XS[i+1]: return YS[i] + SL[i]*(x - XS[i])
def frange(lo, hi):
    v = [f(lo), f(hi)] + [f(x) for x in XS if lo < x < hi]
    return min(v), max(v)

# C' = (1/12)[[11,1],[2,10]] and its inverse
Cm = lambda u, w: (F(11,12)*u + F(1,12)*w, F(1,6)*u + F(5,6)*w)
Ci = lambda y, z: (F(10,9)*y - F(1,9)*z, -F(2,9)*y + F(11,9)*z)

print("=== Proposition 3.1: the local map f")
ok(SL == [-35, F(-5,4), F(5,4), F(7,4), F(-11,6), F(-29,5)], "slopes " + ", ".join(map(s, SL)) + "; all of absolute value >= 5/4")
ok([f(x) for x in XS] == [1, F(1,8), 0, F(1,8), 1, F(29,40), 0] and all(f(x) in XS for x in XS), "f(D) is contained in D (Markov map)")
I = [(XS[j], XS[j+1]) for j in range(6)]
cover = lambda lo, hi: {j for j in range(6) if lo <= I[j][0] and I[j][1] <= hi}
img = [cover(*frange(*I[j])) for j in range(6)]
ok(img == [{2,3,4,5}, {0,1}, {0,1}, {2,3,4,5}, {4,5}, {0,1,2,3}], "f(I1)=f(I4)=I3..I6, f(I2)=f(I3)=I1 u I2, f(I5)=I5 u I6, f(I6)=I1..I4")
ok(all(set().union(*[img[k] for k in img[j]]) == set(range(6)) for j in range(6)), "f^2(I_j) = [0,1] for every j")

def nullspace(M):
    """Basis of the kernel of a matrix of Fractions (Gauss-Jordan elimination)."""
    M = [row[:] for row in M]; n, m = len(M), len(M[0]); piv = []; r = 0
    for c in range(m):
        p = next((i for i in range(r, n) if M[i][c] != 0), None)
        if p is None: continue
        M[r], M[p] = M[p], M[r]; M[r] = [x/M[r][c] for x in M[r]]
        for i in range(n):
            if i != r and M[i][c] != 0: M[i] = [a - M[i][c]*b for a, b in zip(M[i], M[r])]
        piv.append(c); r += 1
    basis = []
    for c in (c for c in range(m) if c not in piv):
        v = [F(0)]*m; v[c] = F(1)
        for i, pc in enumerate(piv): v[pc] = -M[i][c]
        basis.append(v)
    return basis

# transfer operator on step functions: (Lh)_j = sum over i with I_j in f(I_i) of h_i / |slope_i|
Lmat = [[(1/abs(SL[i]) if j in img[i] else F(0)) for i in range(6)] for j in range(6)]
ns = nullspace([[Lmat[j][i] - (i == j) for i in range(6)] for j in range(6)])
h = [x / sum(ns[0][j]*(I[j][1]-I[j][0]) for j in range(6)) for x in ns[0]]
ok(len(ns) == 1 and h == [F(10750,3421)]*2 + [F(2000,3421)]*2 + [F(290,311)]*2,
   "unique invariant step density: 10750/3421 on [0,1/8], 2000/3421 on [1/8,29/40], 290/311 on [29/40,1]")
h1, h2, h3 = F(10750,3421), F(2000,3421), F(290,311)
ok(h1 == F(4,5)*h1 + F(4,5)*h2 + F(5,29)*h3 and h2 == h1/35 + F(4,7)*h2 + F(5,29)*h3
   and h3 == h1/35 + F(4,7)*h2 + F(6,11)*h3 and F(1,8)*h1 + F(3,5)*h2 + F(11,40)*h3 == 1,
   "invariance equations and normalization of Appendix A.2")
lam = sum(float(h[j]*(I[j][1]-I[j][0]))*math.log(abs(float(SL[j]))) for j in range(6))
print(f"     Lyapunov exponent of f: {lam:.4f}; transverse exponent of the diagonal log(3/4) + {lam:.4f} = {math.log(0.75)+lam:.4f} > 0")

print("=== Theorem 3.2 (i): trapping region U = P x P x Q")
P, Q = (F(1,40), F(9,40)), (F(11,20), F(7,8))
fP, fQ = frange(*P), frange(*Q)
m = ((2*fP[0] + fQ[0])/3, (2*fP[1] + fQ[1])/3)
blk = (F(3,4)*fP[0] + m[0]/4, F(3,4)*fP[1] + m[1]/4); sng = (F(3,4)*fQ[0] + m[0]/4, F(3,4)*fQ[1] + m[1]/4)
ok(fP == (0, F(1,8)) and fQ == (F(111,160), 1), f"f(P) = [{fP[0]},{fP[1]}], f(Q) = [{fQ[0]},{fQ[1]}]")
ok(m == (F(37,160), F(5,12)), f"mean field in [{m[0]},{m[1]}]")
ok(blk == (F(37,640), F(19,96)) and sng == (F(37,64), F(41,48)), f"x1', x2' in [{blk[0]},{blk[1]}], x3' in [{sng[0]},{sng[1]}]")
marg = min(blk[0]-P[0], P[1]-blk[1], sng[0]-Q[0], Q[1]-sng[1])
ok(marg == F(1,48), f"F(U) in int U, with margin {marg}")
ok([x for x in XS if P[0] < x < P[1] or Q[0] < x < Q[1]] == [F(1,8), F(29,40)], "the only breakpoints in int P u int Q are 1/8 and 29/40")
ok(all(abs(SL[j]) == F(5,4) for j in (1, 2)), "|f'| = 5/4 on P, so the pair contracts by (3/4)(5/4) = 15/16 (Theorem 3.2 (iii))")

print("=== Theorem 3.2 (iv), Lemma A.1: the invariant parallelogram S = C'(B_S)")
BS = (F(0), F(9,160), F(7,8), F(23,24)); J = (F(2,25), F(1,8)); K = (F(329,440), F(349,440))
ok(sorted([f(J[0]), f(J[1])]) == [BS[0], BS[1]] and J[1] <= F(1,8), "J = [2/25,1/8] lies in the left lap of f|P, which maps it onto [0,9/160]")
ok(sorted([f(K[0]), f(K[1])]) == [BS[2], BS[3]] and K[0] >= F(29,40), "K = [329/440,349/440] lies in the right lap of f|Q, which maps it onto [7/8,23/24]")
corners = [Ci(y, z) for y in J for z in K]
ok(all(BS[0] < u < BS[1] and BS[2] < w < BS[3] for u, w in corners),
   "C'^-1(corners of J x K) in int B_S: " + ", ".join(f"({s(u)},{s(w)})" for u, w in corners))
V = [Cm(u, w) for u, w in ((BS[0], BS[2]), (BS[1], BS[2]), (BS[1], BS[3]), (BS[0], BS[3]))]
ok(all(P[0] <= y <= P[1] and Q[0] <= z <= Q[1] for y, z in V), "vertices of S, all in R: " + ", ".join(f"({s(y)},{s(z)})" for y, z in V))
aS = F(3,4)*(BS[1]-BS[0])*(BS[3]-BS[2])
ok(aS == F(9,2560), f"area(S) = det(C') area(B_S) = {aS}")

print("=== Proposition 3.4: uniform expansion of G = C'(f x f) on the four rectangles")
ok(F(3,4)*F(5,4)*F(7,4) == F(105,64) and F(3,4)*F(5,4)*F(11,6) == F(55,32), "|det DG| = 105/64 or 55/32")
def expansion(b, tr_, det_, dd_):
    M = [[F(11,12)*F(5,4), F(1,12)*b], [F(1,6)*F(5,4), F(5,6)*b]]               # M_b = C' diag(5/4, b)
    MtM = [[M[0][i]*M[0][j] + M[1][i]*M[1][j] for j in range(2)] for i in range(2)]
    tr = MtM[0][0] + MtM[1][1]; det = MtM[0][0]*MtM[1][1] - MtM[0][1]**2; dd = det - tr + 1   # det(MtM - I)
    sig = math.sqrt((float(tr) - math.sqrt(float(tr)**2 - 4*float(det)))/2)
    ok(tr == tr_ and det == det_ and dd == dd_ and tr > 2 and dd > 0,
       f"b = {b}: tr(M^T M) = {tr} > 2, det(M^T M - I) = {dd} > 0, so sigma_min(M_b) = {sig:.4f} > 1")
    return sig
sigma = min(expansion(F(7,4), F(4037,1152), F(11025,4096), F(6905,36864)),
            expansion(F(11,6), F(77009,20736), F(3025,1024), F(19933,82944)))
print(f"     sigma = {sigma:.4f} (DG = M_b diag(+-1, +-1), with the same singular values as M_b)")

print("=== Proposition 3.3: exact attractor G^7(R) = G^6(R)")
GR = [L.Cmul(p) for p in [(fP[0], fQ[0]), (fP[1], fQ[0]), (fP[1], fQ[1]), (fP[0], fQ[1])]]
areas = {1: L.area(GR)}; level = [GR]
for n in range(2, 8):
    level = [[L.Cmul(p) for p in poly] for poly in L.fxf_image(level)]   # G^n(R) as a union of convex polygons
    areas[n] = L.union_area(level)
    print(f"     n = {n}: {len(level):4d} convex pieces, area G^{n}(R) = {float(areas[n]):.9f}", flush=True)
A = F(25751997557087, 2614053856870400)
ok(areas[1] == F(147,5120), f"area G(R) = {areas[1]}")
ok(areas[7] == areas[6] == A, f"area G^7(R) = area G^6(R) = {A} (about {float(A):.5f}); since G^7(R) is contained in G^6(R), they are equal")
print(f"     area(A_red)/area(G(R)) = {float(A/areas[1]):.4f}; area(S)/area(A_red) = {float(aS/A):.4f}")

print("\nALL CHECKS PASSED" if not FAILS else f"\n{len(FAILS)} CHECK(S) FAILED")
sys.exit(1 if FAILS else 0)
