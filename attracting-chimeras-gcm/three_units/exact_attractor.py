"""Exact iteration of G on R = P x Q (Proposition 3.3): G^n(R) as unions of convex polygons with rational vertices.

Iterates until the area stabilizes (n = 7) and saves the polygons of G^n(R), n <= 7, to exact_levels.pkl,
which is used by occupancy.py. (verify_exact.py performs the same computation and checks the stated numbers.)
Standard library only; about 20 s.
"""
from fractions import Fraction as Fr
import pickle, time
from exact_polygons import Cmul, fxf_image, union_area, area

GR = [Cmul(p) for p in [(Fr(0), Fr(111,160)), (Fr(1,8), Fr(111,160)), (Fr(1,8), Fr(1)), (Fr(0), Fr(1))]]   # G(R) = C'(f(P) x f(Q))
print("area G(R) =", area(GR))
level = [GR]; prev = None; t0 = time.time(); hist = {1: [GR]}
for n in range(2, 12):
    level = [[Cmul(p) for p in poly] for poly in fxf_image(level)]; aG = union_area(level); hist[n] = level
    print(f"n={n}: {len(level)} convex pieces, area G^{n}(R) = {aG} = {float(aG):.12f} ({time.time()-t0:.0f}s)", flush=True)
    if aG == prev: print(f"   area of G^{n}(R) equals that of G^{n-1}(R): the attractor is G^{n-1}(R)"); break
    prev = aG
pickle.dump(hist, open("exact_levels.pkl", "wb"))
print("saved exact_levels.pkl")
