"""
Exact verification of the four-unit example (Section 3.1, Theorem 3.6; proof in Appendix A.10).

N = 4, eps = 1/4, local map f^ = piecewise linear interpolation of
(0,1), (1/30,1/6), (1/6,0), (3/10,1/6), (4/5,1), (23/25,4/5), (1,0);
P^ = [1/30,3/10] (synchronized pair), Q^ = [17/25,23/25] (two free units), U^ = P^ x P^ x Q^ x Q^.
Checks the Markov structure and invariant density of f^ (i), the trapping region and the transverse
contraction (ii), the uniform expansion of the reduced map G^ (iii), the ball B in the attractor (iv),
and the multiplier 5/4 of the pair of free units (v). Only the Python standard library is needed.
"""
import sys
from fractions import Fraction as Fr
import itertools, math

FAILS = []
def ok(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond: FAILS.append(msg)
s = lambda x: str(x)
e = Fr(1,4)

pts = [(Fr(0),Fr(1)), (Fr(1,30),Fr(1,6)), (Fr(1,6),Fr(0)), (Fr(3,10),Fr(1,6)), (Fr(4,5),Fr(1)), (Fr(23,25),Fr(4,5)), (Fr(1),Fr(0))]
D = [p[0] for p in pts]
SL = [(fb-fa)/(b-a) for (a,fa), (b,fb) in zip(pts, pts[1:])]
def f(x):
    for (a,fa), (b,fb) in zip(pts, pts[1:]):
        if a <= x <= b: return fa + (fb-fa)*(x-a)/(b-a)
def frange(lo, hi):
    v = [f(x) for x in [lo, hi] + [a for a in D if lo < a < hi]]
    return min(v), max(v)

print("=== (i) the local map")
ok(SL == [-25, Fr(-5,4), Fr(5,4), Fr(5,3), Fr(-5,3), -10] and min(abs(x) for x in SL) == Fr(5,4),
   "slopes " + ", ".join(map(s, SL)) + "; |f^'| >= 5/4")
ok([f(x) for x in D] == [1, Fr(1,6), 0, Fr(1,6), 1, Fr(4,5), 0], "breakpoints mapped to 1, 1/6, 0, 1/6, 1, 4/5, 0 (Markov map)")
I = list(zip(D, D[1:]))
img = [{k for k, (u, w) in enumerate(I) if frange(*J)[0] <= u and w <= frange(*J)[1]} for J in I]
ok(img == [{2,3,4,5}, {0,1}, {0,1}, {2,3,4,5}, {4,5}, {0,1,2,3}], "same transitions as the map f of the main example")
ok(all(set().union(*[img[k] for k in img[j]]) == set(range(6)) for j in range(6)), "f^^2(I_j) = [0,1] for every j")
# invariant density h1, h2, h3 on [0,1/6), [1/6,4/5), [4/5,1]: transfer operator on step functions
h1, h2, h3 = Fr(75,28), Fr(15,28), Fr(15,14)
H = [h1, h1, h2, h2, h3, h3]
Lh = [sum(H[i]/abs(SL[i]) for i in range(6) if j in img[i]) for j in range(6)]
ok(Lh == H and sum(H[j]*(b-a) for j, (a, b) in enumerate(I)) == 1, "invariant density 75/28, 15/28, 15/14 (normalized)")
lyap = sum(float(H[j]*(b-a))*math.log(abs(float(SL[j]))) for j, (a, b) in enumerate(I))
print(f"     Lyapunov exponent of f^: {lyap:.4f}; transverse exponent of the diagonal log(3/4) + {lyap:.4f} = {math.log(0.75)+lyap:.4f} > 0")

print("=== (ii) trapping region and transverse contraction")
P, Q = (Fr(1,30), Fr(3,10)), (Fr(17,25), Fr(23,25))
fP, fQ = frange(*P), frange(*Q)
mf = ((2*fP[0] + 2*fQ[0])/4, (2*fP[1] + 2*fQ[1])/4)
blk = ((1-e)*fP[0] + e*mf[0], (1-e)*fP[1] + e*mf[1]); fr = ((1-e)*fQ[0] + e*mf[0], (1-e)*fQ[1] + e*mf[1])
ok(fP == (0, Fr(1,6)) and fQ == (Fr(4,5), 1), f"f^(P^) = [{fP[0]},{fP[1]}], f^(Q^) = [{fQ[0]},{fQ[1]}]")
ok(mf == (Fr(2,5), Fr(7,12)), f"mean field in [{mf[0]},{mf[1]}]")
ok(blk == (Fr(1,10), Fr(13,48)) and fr == (Fr(7,10), Fr(43,48)), f"x1', x2' in [{blk[0]},{blk[1]}], x3', x4' in [{fr[0]},{fr[1]}]")
marg = min(blk[0]-P[0], P[1]-blk[1], fr[0]-Q[0], Q[1]-fr[1])
ok(marg > 0, f"F^(U^) in int U^, with margin {marg}")
ok([a for a in D if P[0] < a < P[1]] == [Fr(1,6)] and abs(SL[1]) == abs(SL[2]) == Fr(5,4),
   "|f^'| = 5/4 on P^: the pair contracts by (3/4)(5/4) = 15/16, TLE log(15/16)")

print("=== (iii) uniform expansion of G^ = C^ (f^ x f^ x f^)")
C = [[1-e/2, e/4, e/4], [e/2, 1-3*e/4, e/4], [e/2, e/4, 1-3*e/4]]
ok(C == [[Fr(14,16), Fr(1,16), Fr(1,16)], [Fr(2,16), Fr(13,16), Fr(1,16)], [Fr(2,16), Fr(1,16), Fr(13,16)]], "C^ = (1/16)[[14,1,1],[2,13,1],[2,1,13]]")
mm = lambda A, B: [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]
def det(A):
    if len(A) == 1: return A[0][0]
    return sum((-1)**j*A[0][j]*det([r[:j] + r[j+1:] for r in A[1:]]) for j in range(len(A)))
M = mm(C, [[Fr(5,4),0,0], [0,Fr(5,3),0], [0,0,Fr(5,3)]])
ok(M == [[Fr(35,32), Fr(5,48), Fr(5,48)], [Fr(5,32), Fr(65,48), Fr(5,48)], [Fr(5,32), Fr(5,48), Fr(65,48)]],
   "M^ = C^ diag(5/4,5/3,5/3) = [[35/32,5/48,5/48],[5/32,65/48,5/48],[5/32,5/48,65/48]]")
K = [[x - (i == j) for j, x in enumerate(row)] for i, row in enumerate(mm([list(r) for r in zip(*M)], M))]
minors = [det([r[:k] for r in K[:k]]) for k in (1, 2, 3)]
ok(minors == [Fr(251,1024), Fr(3043,32768), Fr(441,16384)] and all(x > 0 for x in minors),
   "leading principal minors of M^T M - I: " + ", ".join(map(s, minors)) + " > 0, so sigma_min(M^) > 1")
# smallest singular value: smallest eigenvalue of M^T M, by bisection on the characteristic polynomial
MtM = mm([list(r) for r in zip(*M)], M)
charp = lambda t: float(det([[x - t*(i == j) for j, x in enumerate(row)] for i, row in enumerate(MtM)]))
lo, hi = 1.0, 1.1
for _ in range(60):
    mid = (lo + hi)/2; lo, hi = (mid, hi) if charp(mid)*charp(lo) > 0 else (lo, mid)
print(f"     sigma^ = sqrt(lambda_min(M^T M)) = {math.sqrt(lo):.4f}")

print("=== (iv) a ball in the attractor")
laps = {'vl': (Fr(-5,4), Fr(5,24), (P[0], Fr(1,6))), 'vr': (Fr(5,4), Fr(-5,24), (Fr(1,6), P[1])),
        'pl': (Fr(5,3), Fr(-1,3), (Q[0], Fr(4,5))), 'pr': (Fr(-5,3), Fr(7,3), (Fr(4,5), Q[1]))}
ok(all(sl*a + c == f(a) and sl*b + c == f(b) for sl, c, (a, b) in laps.values()), "affine laps of f^ on P^ and Q^")
def fixed(box):
    sl = [laps[k][0] for k in box]; t = [laps[k][1] for k in box]
    A = [[(i == j) - C[i][j]*sl[j] for j in range(3)] for i in range(3)]; rhs = [sum(C[i][j]*t[j] for j in range(3)) for i in range(3)]
    d = det(A); p = []
    for k in range(3):
        Ak = [r[:] for r in A]
        for i in range(3): Ak[i][k] = rhs[i]
        p.append(det(Ak)/d)
    return p
box = ('vl', 'pr', 'pr')
p = fixed(box)
dist = [min(x - laps[k][2][0], laps[k][2][1] - x) for k, x in zip(box, p)]
ok(p == [Fr(141,982), Fr(817,982), Fr(817,982)], "fixed point of the affine branch (left lap of the valley, right lap of the peak twice): p = (141/982, 817/982, 817/982)")
ok(dist[0] == Fr(34,1473) and min(dist) == Fr(34,1473), f"distance from p to the faces of its box: {', '.join(map(s, dist))}; minimum 34/1473 (face y = 1/6)")
print("     (the ball B of radius 34/1473 around p lies in the box and B is contained in G^(B), since G^(B) is an ellipsoid")
print("      centered at p with semi-axes at least sigma^ * 34/1473)")

print("=== (v) the free units do not synchronize")
ok((1-e)*Fr(5,3) == Fr(5,4), "x3' - x4' = (3/4)(f^(x3) - f^(x4)) and |f^'| = 5/3 on Q^: the cluster space {x3 = x4} has TLE log(5/4) > 0")

print("\nALL CHECKS PASSED" if not FAILS else f"\n{len(FAILS)} CHECK(S) FAILED")
sys.exit(1 if FAILS else 0)
