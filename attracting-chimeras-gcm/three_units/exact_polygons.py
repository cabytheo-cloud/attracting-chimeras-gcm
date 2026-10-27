"""
Exact polygon tools for the main example (six-piece map f, N = 3, eps = 1/4).

G = C'(f x f) is affine on each of the four rectangles of R = P x Q formed by the two laps of f on P
and the two laps of f on Q (the lists Pp, Qp below). The functions compute images of unions of convex
polygons under f x f and exact areas of such unions by a vertical sweep. All data are rational, so all
computations are exact (fractions.Fraction). Used by verify_exact.py and exact_attractor.py to show
that G^7(R) = G^6(R) (Proposition 3.3 and Appendix A.7 of the paper).
"""
from fractions import Fraction as Fr
from itertools import combinations

C = ((Fr(11, 12), Fr(1, 12)), (Fr(1, 6), Fr(5, 6)))
def Cmul(p):
    return (C[0][0]*p[0] + C[0][1]*p[1], C[1][0]*p[0] + C[1][1]*p[1])

# pieces: (y0,y1,sy,cy), (z0,z1,sz,cz) with f(y)=sy*y+cy, f(z)=sz*z+cz
Pp = [(Fr(1,40), Fr(1,8), Fr(-5,4), Fr(5,32)), (Fr(1,8), Fr(9,40), Fr(5,4), Fr(-5,32))]
Qp = [(Fr(11,20), Fr(29,40), Fr(7,4), Fr(-43,160)), (Fr(29,40), Fr(7,8), Fr(-11,6), Fr(559,240))]

def clip(poly, axis, val, keep_ge):
    """Sutherland-Hodgman clip of convex polygon against half-plane coord[axis] >= val (or <=)."""
    out = []
    n = len(poly)
    inside = (lambda p: p[axis] >= val) if keep_ge else (lambda p: p[axis] <= val)
    for i in range(n):
        P, Qv = poly[i], poly[(i+1) % n]
        if inside(P):
            out.append(P)
        if inside(P) != inside(Qv):
            t = (val - P[axis]) / (Qv[axis] - P[axis])
            out.append((P[0] + t*(Qv[0]-P[0]), P[1] + t*(Qv[1]-P[1])))
    # remove consecutive duplicates
    res = []
    for p in out:
        if not res or res[-1] != p:
            res.append(p)
    if len(res) > 1 and res[0] == res[-1]:
        res.pop()
    return res

def area(poly):
    s = 0
    for i in range(len(poly)):
        x1, y1 = poly[i]; x2, y2 = poly[(i+1) % len(poly)]
        s += x1*y2 - x2*y1
    return abs(s) / 2

def fxf_image(convex_polys):
    """Image under f x f of a union of convex polygons contained in P x Q: list of convex polygons."""
    out = []
    for poly in convex_polys:
        for (y0, y1, sy, cy) in Pp:
            for (z0, z1, sz, cz) in Qp:
                q = clip(poly, 0, y0, True)
                if len(q) >= 3: q = clip(q, 0, y1, False)
                if len(q) >= 3: q = clip(q, 1, z0, True)
                if len(q) >= 3: q = clip(q, 1, z1, False)
                if len(q) >= 3 and area(q) > 0:
                    out.append([(sy*p[0] + cy, sz*p[1] + cz) for p in q])
    return out

def slice_at(poly, u):
    """vertical slice [lo,hi] of a convex polygon at x=u, or None."""
    ys = []
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i+1) % n]
        if x1 == x2:
            if x1 == u: ys += [y1, y2]
            continue
        if min(x1, x2) <= u <= max(x1, x2):
            t = (u - x1) / (x2 - x1)
            ys.append(y1 + t*(y2 - y1))
    if not ys: return None
    return (min(ys), max(ys))

def union_area(polys):
    """Exact area of a union of convex polygons by an exact vertical sweep."""
    crit = set()
    edges = []
    for k, poly in enumerate(polys):
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i+1) % n]
            crit.add(a[0])
            if a[0] != b[0]:
                edges.append((k, a, b))
    # edge-edge crossings (different polygons) change the order of slice endpoints
    for (k1, a1, b1), (k2, a2, b2) in combinations(edges, 2):
        if k1 == k2: continue
        lo = max(min(a1[0], b1[0]), min(a2[0], b2[0])); hi = min(max(a1[0], b1[0]), max(a2[0], b2[0]))
        if lo >= hi: continue
        m1 = (b1[1]-a1[1])/(b1[0]-a1[0]); m2 = (b2[1]-a2[1])/(b2[0]-a2[0])
        if m1 == m2: continue
        u = (a2[1] - m2*a2[0] - a1[1] + m1*a1[0]) / (m1 - m2)
        if lo < u < hi: crit.add(u)
    xs = sorted(crit)
    total = Fr(0)
    for u0, u1 in zip(xs, xs[1:]):
        um = (u0 + u1) / 2
        ivs = sorted(s for s in (slice_at(p, um) for p in polys) if s is not None)
        L = Fr(0); cur = None
        for lo, hi in ivs:
            if cur is None or lo > cur[1]:
                if cur: L += cur[1] - cur[0]
                cur = [lo, hi]
            else:
                cur[1] = max(cur[1], hi)
        if cur: L += cur[1] - cur[0]
        total += L * (u1 - u0)          # union length is linear on [u0,u1]: midpoint rule exact
    return total

