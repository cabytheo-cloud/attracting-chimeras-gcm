"""
Second stage of the search that produced the six-piece map f: for the best candidates found by search.py
(saved in cands.pkl), run the reduced map and test whether the attractor of the chimera forms one piece or splits
into bands. Usage: python merge_test.py cands.pkl 25
"""
import numpy as np, pickle, sys
from numba import njit
@njit
def run(XS, YS, y, z, T0, T):
    n = XS.size - 1
    Y = np.empty(T); Z = np.empty(T)
    for t in range(T0 + T):
        fy = YS[-1]; fz = YS[-1]
        for i in range(n):
            if y <= XS[i+1]:
                fy = YS[i] + (YS[i+1]-YS[i])*(y-XS[i])/(XS[i+1]-XS[i]); break
        for i in range(n):
            if z <= XS[i+1]:
                fz = YS[i] + (YS[i+1]-YS[i])*(z-XS[i])/(XS[i+1]-XS[i]); break
        y, z = 11/12*fy+1/12*fz, fy/6+5/6*fz
        if t >= T0: Y[t-T0] = y; Z[t-T0] = z
    return Y, Z
def test(r, T=2_000_000):
    XS = np.array([float(x) for x in r["XS"]]); YS = np.array([float(y) for y in r["YS"]])
    y0 = (r["P"][0]+r["P"][1])/2; z0 = (r["Q"][0]+r["Q"][1])/2
    Y, Z = run(XS, YS, y0, z0, 3000, T)
    ys = np.sort(Y); gy = np.diff(ys).max(); zs = np.sort(Z); gz = np.diff(zs).max()
    H, _, _ = np.histogram2d(Y, Z, bins=200, range=[[ys[0], ys[-1]], [zs[0], zs[-1]]])
    occ = (H > 0).sum()*(ys[-1]-ys[0])*(zs[-1]-zs[0])/200**2
    # two-band test: transitions between y below/above the largest gap
    i = np.diff(ys).argmax(); cut = (ys[i]+ys[i+1])/2; lab = (Y > cut).astype(int)
    stay = np.mean(lab[1:] == lab[:-1])
    inP = np.all((Y > r["P"][0]) & (Y < r["P"][1])); inQ = np.all((Z > r["Q"][0]) & (Z < r["Q"][1]))
    return dict(gapy=gy, gapz=gz, occ=occ, stay=stay, inPQ=inP and inQ, yr=(ys[0], ys[-1]), zr=(zs[0], zs[-1]))
if __name__ == "__main__":
    res = pickle.load(open(sys.argv[1], "rb"))
    res.sort(key=lambda r: -min(r["margin"], (r["sigma"] - 1)/3))
    for r in res[:int(sys.argv[2])]:
        t = test(r, 400_000)
        print(f"margin {r['margin']:.4f} sigma {r['sigma']:.4f} ratio {r['ratio']:.2f} | gap_y {t['gapy']:.4f} stay {t['stay']:.3f} occ {t['occ']:.4f} | XS {[str(x) for x in r['XS']]} YS {[str(y) for y in r['YS']]}", flush=True)
