"""
Numbers used in the proof of Proposition 3.5 (i) (robustness under C^0 perturbations; Appendix A.9).

  - trapping margins for the enlarged region U' = P' x P' x Q, P' = [1/20, 1/5]: 1/128 and 1/480 (> 2 eta, eta = 6e-5);
  - d = dist(D, boundary of S) > 3.9e-4, where D = J x K is mapped by G affinely onto the parallelogram S;
  - S is contained in R' = P' x Q;
  - operator norms ||C'_eps - C'|| < 1.06 |eps - 1/4| and ||C'_eps|| < 1.01 for |eps - 1/4| <= 6e-5;
  - the smooth example g = f * rho, w = 1e-5: |g - f| <= (29/5) w < eta on P' u Q, and |g'| <= 5/4 on P'.
Exact arithmetic except for the square roots in distances and norms. Needs numpy.
"""
from fractions import Fraction as F
import math
import numpy as np

def ok(cond, msg): print(("OK   " if cond else "FAIL ") + msg)

XS = [F(0), F(1,40), F(1,8), F(9,40), F(29,40), F(7,8), F(1)]
YS = [F(1), F(1,8), F(0), F(1,8), F(1), F(29,40), F(0)]
SL = [(YS[i+1]-YS[i])/(XS[i+1]-XS[i]) for i in range(6)]
def f(x):
    for i in range(6):
        if XS[i] <= x <= XS[i+1]: return YS[i] + SL[i]*(x - XS[i])
Cm = lambda p: (F(11,12)*p[0] + F(1,12)*p[1], F(1,6)*p[0] + F(5,6)*p[1])
eta, w = F(6, 100000), F(1, 100000)
P1, Q = (F(1,20), F(1,5)), (F(11,20), F(7,8))

# trapping margins for U' (images of U' under F_eps lie in the images of U computed in Appendix A.3)
blk, sng = (F(37,640), F(19,96)), (F(37,64), F(41,48))
ok(F(1,40) <= P1[0] and P1[1] <= F(9,40), "P' = [1/20,1/5] is contained in P = [1/40,9/40], so U' is contained in U")
m1, m2 = blk[0] - P1[0], P1[1] - blk[1]
ok(m1 == F(1,128) and m2 == F(1,480) and min(m1, m2, sng[0]-Q[0], Q[1]-sng[1]) == F(1,480) > 2*eta,
   f"margins of F_eps(U') in U': {m1}, {m2} (and >= 1/48 in Q); minimum 1/480 > 2 eta = {2*eta}")

# the parallelogram S = C'(B_S) and D = J x K, with G(D) = S
BS = [(F(0), F(7,8)), (F(9,160), F(7,8)), (F(9,160), F(23,24)), (F(0), F(23,24))]
S = [Cm(p) for p in BS]
J, K = (F(2,25), F(1,8)), (F(329,440), F(349,440))
D = [(J[0], K[0]), (J[1], K[0]), (J[1], K[1]), (J[0], K[1])]
G = lambda p: Cm((f(p[0]), f(p[1])))
ok(sorted(G(p) for p in D) == sorted(S), "G maps the corners of D = J x K onto the vertices of S")
def signed_dist(p, a, b):
    cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
    return cross, float(cross)/math.hypot(float(b[0]-a[0]), float(b[1]-a[1]))
sd = [signed_dist(p, S[i], S[(i+1) % 4]) for p in D for i in range(4)]
d = min(abs(x[1]) for x in sd)
ok(len({x[0] > 0 for x in sd}) == 1, "every corner of D lies strictly on the same side of every edge of S, so D is in int S")
ok(d > 3.9e-4, f"d = dist(D, boundary of S) = {d:.6g} > 3.9e-4, and 3 eta = {3*float(eta):.1e} < d/2")
ok(all(P1[0] <= y <= P1[1] and Q[0] <= z <= Q[1] for y, z in S), "S is contained in R' = P' x Q")

# operator norms
Ce = lambda e: np.array([[1 - e/3, e/3], [2*e/3, 1 - 2*e/3]])
nd = np.linalg.norm(np.array([[-1/3, 1/3], [2/3, -2/3]]), 2)
nC = max(np.linalg.norm(Ce(e), 2) for e in np.linspace(0.25 - 6e-5, 0.25 + 6e-5, 101))
ok(nd < 1.06 and nC < 1.01, f"||C'_eps - C'|| = {nd:.4f} |eps - 1/4| and ||C'_eps|| <= {nC:.4f} for |eps - 1/4| <= 6e-5")
ok(1.06*math.sqrt(2) < 1.5 and nC*math.sqrt(2) < 1.5, "hence |G_g - G| < (3/2)|eps - 1/4| + (3/2) eta on D (|(f(y), f(z))| <= sqrt 2)")

# smooth example g = f * rho, rho supported in [-w, w]
near = lambda lo, hi: [i for i in range(6) if XS[i+1] > lo - w and XS[i] < hi + w]   # pieces meeting the w-neighbourhood
pieces = sorted(set(near(*P1)) | set(near(*Q)))
lip = max(abs(SL[i]) for i in pieces)
ok(lip == F(29,5) and lip*w < eta, f"slopes of the pieces within distance w of P' u Q are at most {lip}, so |g - f| <= {lip*w} < eta")
ok([x for x in XS if P1[0] - w < x < P1[1] + w] == [F(1,8)] and all(abs(SL[i]) == F(5,4) for i in near(*P1)),
   "1/8 is the only breakpoint within distance w of P', and |f'| = 5/4 there, so |g'| <= 5/4 on P'")
