"""
Exact verification of the four-piece example of Section 6.2 (Theorem 6.1, proof in Appendix A.12).

N = 3, eps = 1/4, local map f~ = piecewise linear interpolation of (0,0), (1/8,5/32), (1/4,0), (3/4,1), (1,1/2),
P~ = [0,1/4], Q~ = [19/30,13/15], U~ = P~ x P~ x Q~, R~ = P~ x Q~, G~ = C'(f~ x f~).
Checks the trapping region, the transverse contraction, the invariant parallelogram, the uniform
expansion and the exact attractor G~^4(R~) = G~^3(R~). Only the Python standard library is needed (a few seconds).
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

XS = [F(0), F(1,8), F(1,4), F(3,4), F(1)]
YS = [F(0), F(5,32), F(0), F(1), F(1,2)]
SL = [(YS[i+1]-YS[i])/(XS[i+1]-XS[i]) for i in range(4)]
def f(x):
    for i in range(4):
        if XS[i] <= x <= XS[i+1]: return YS[i] + SL[i]*(x - XS[i])
def frange(lo, hi):
    v = [f(lo), f(hi)] + [f(x) for x in XS if lo < x < hi]
    return min(v), max(v)
Cm = lambda u, w: (F(11,12)*u + F(1,12)*w, F(1,6)*u + F(5,6)*w)
Ci = lambda y, z: (F(10,9)*y - F(1,9)*z, -F(2,9)*y + F(11,9)*z)

print("=== the local map f~")
ok(SL == [F(5,4), F(-5,4), 2, -2], "slopes 5/4, -5/4 (small tent on the invariant interval [0,1/4]) and 2, -2 (large tent)")
ok(frange(F(0), F(1,4)) == (0, F(5,32)), "f~([0,1/4]) = [0,5/32], so [0,1/4] is invariant and f~ is not ergodic")

print("=== trapping region U~ (Theorem 3.2 (i) for f~)")
P, Q = (F(0), F(1,4)), (F(19,30), F(13,15))
fP, fQ = frange(*P), frange(*Q)
m = ((2*fP[0] + fQ[0])/3, (2*fP[1] + fQ[1])/3)
blk = (F(3,4)*fP[0] + m[0]/4, F(3,4)*fP[1] + m[1]/4); sng = (F(3,4)*fQ[0] + m[0]/4, F(3,4)*fQ[1] + m[1]/4)
ok(fP == (0, F(5,32)) and fQ == (F(23,30), 1), f"f~(P~) = [{fP[0]},{fP[1]}], f~(Q~) = [{fQ[0]},{fQ[1]}]")
ok(m == (F(23,90), F(7,16)), f"mean field in [{m[0]},{m[1]}]")
ok(blk == (F(23,360), F(29,128)) and sng == (F(23,36), F(55,64)), f"x1', x2' in [{blk[0]},{blk[1]}], x3' in [{sng[0]},{sng[1]}]")
marg = min(blk[0]-P[0], P[1]-blk[1], sng[0]-Q[0], Q[1]-sng[1])
ok(marg == F(1,180), f"F(U~) in int U~, with margin {marg}")
ok(abs(SL[0]) == abs(SL[1]) == F(5,4), "|f~'| = 5/4 on P~: the pair contracts by 15/16 and the TLE is log(15/16)")

print("=== invariant parallelogram S~ (Theorem 3.2 (iv) for f~)")
BS = (F(3,64), F(5,32), F(6,7), F(1)); J = (F(1,8), F(17,80)); K = (F(3,4), F(23,28))
ok(sorted([f(J[0]), f(J[1])]) == [BS[0], BS[1]] and J[0] >= F(1,8), "J~ = [1/8,17/80] lies in the right lap of f~|P~, which maps it onto [3/64,5/32]")
ok(sorted([f(K[0]), f(K[1])]) == [BS[2], BS[3]] and K[0] >= F(3,4), "K~ = [3/4,23/28] lies in the right lap of f~|Q~, which maps it onto [6/7,1]")
corners = [Ci(y, z) for y in J for z in K]
ok(all(BS[0] <= u <= BS[1] and BS[2] <= w <= BS[3] for u, w in corners),
   "C'^-1(corners of J~ x K~) in B~_S: " + ", ".join(f"({s(u)},{s(w)})" for u, w in corners))
V = [Cm(u, w) for u, w in ((BS[0], BS[2]), (BS[1], BS[2]), (BS[1], BS[3]), (BS[0], BS[3]))]
ok(all(P[0] <= y <= P[1] and Q[0] <= z <= Q[1] for y, z in V), "vertices of S~, all in R~: " + ", ".join(f"({s(y)},{s(z)})" for y, z in V))
aS = F(3,4)*(BS[1]-BS[0])*(BS[3]-BS[2])
ok(aS == F(3,256), f"area(S~) = {aS}")

print("=== uniform expansion (Proposition 3.4 for f~)")
M = [[F(11,12)*F(5,4), F(1,12)*2], [F(1,6)*F(5,4), F(5,6)*2]]
MtM = [[M[0][i]*M[0][j] + M[1][i]*M[1][j] for j in range(2)] for i in range(2)]
tr = MtM[0][0] + MtM[1][1]; det = MtM[0][0]*MtM[1][1] - MtM[0][1]**2; dd = det - tr + 1
sig = math.sqrt((float(tr) - math.sqrt(float(tr)**2 - 4*float(det)))/2)
ok(tr == F(9589,2304) and dd == F(815,2304) and tr > 2 and dd > 0,
   f"M = C' diag(5/4, 2): tr(M^T M) = {tr} > 2, det(M^T M - I) = {dd} > 0, sigma_min = {sig:.4f} > 1")

print("=== exact attractor G~^4(R~) = G~^3(R~)")
GR = [L.Cmul(p) for p in [(fP[0], fQ[0]), (fP[1], fQ[0]), (fP[1], fQ[1]), (fP[0], fQ[1])]]
areas = {1: L.area(GR)}; level = [GR]
for n in range(2, 5):
    level = [[L.Cmul(p) for p in poly] for poly in L.fxf_image(level)]
    areas[n] = L.union_area(level)
    print(f"     n = {n}: {len(level):3d} convex pieces, area G~^{n}(R~) = {float(areas[n]):.9f}")
A = F(12939, 720896)
ok(areas[1] == F(7,256), f"area G~(R~) = {areas[1]}")
ok(areas[4] == areas[3] == A, f"area G~^4(R~) = area G~^3(R~) = {A}; since G~^4(R~) is contained in G~^3(R~), they are equal")

print("\nALL CHECKS PASSED" if not FAILS else f"\n{len(FAILS)} CHECK(S) FAILED")
sys.exit(1 if FAILS else 0)
