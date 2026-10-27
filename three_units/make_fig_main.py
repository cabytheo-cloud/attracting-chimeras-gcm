"""
Figure 1 of the paper (fig_main.pdf): (a) the local map f, (b) its invariant density (exact and histogram),
(c) the empirical invariant density of G on the attractor A_red, with the parallelogram S.
Needs density_hist.npz (run density_hist.py first), numpy, numba and matplotlib.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from numba import njit
from common import f, G, P, Q, XS, YS
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
Cp = np.array([[11/12, 1/12], [1/6, 5/6]])
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
@njit(cache=True)
def hist1(x, n, nb):
    H = np.zeros(nb)
    for _ in range(1000): x = f(x)
    for _ in range(n):
        x = f(x); H[min(int(x*nb), nb-1)] += 1
    return H
nb = 400; H1 = sum(hist1(x0, 50_000_000, nb) for x0 in (0.11, 0.53, 0.97)); D1 = H1/H1.sum()*nb
dh = np.load("density_hist.npz"); D = dh["D"].astype(float); y0, z0, h = float(dh["y0"]), float(dh["z0"]), float(dh["h"])
D[D == 0] = np.nan; ext = [y0, y0 + h*D.shape[0], z0, z0 + h*D.shape[1]]
S = np.array([Cp @ p for p in [(0, 7/8), (9/160, 7/8), (9/160, 23/24), (0, 23/24)]])
fig = plt.figure(figsize=(11, 7.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], wspace=0.25, hspace=0.38)
a = fig.add_subplot(gs[0, 0])
a.axvspan(*P, color=BLUE, alpha=0.12, lw=0); a.axvspan(*Q, color=ORANGE, alpha=0.12, lw=0)
a.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=(0, (4, 3))); a.plot(XS, YS, color=INK, lw=1.7, marker="o", ms=3.5)
a.text(0.125, 0.60, "P\nblock\n|f′| = 5/4", ha="center", va="top", fontsize=9, color=INK2)
a.text(0.7125, 0.52, "Q\nsingleton\n|f′| = 7/4, 11/6", ha="center", va="top", fontsize=9, color=INK2)
a.set_xlim(0, 1); a.set_ylim(0, 1.05); a.set_xlabel("x"); a.set_ylabel("f(x)")
a.set_title("(a) Local map f: 6 linear pieces", loc="left", fontsize=12); clean(a)
a = fig.add_subplot(gs[1, 0])
xc = (np.arange(nb)+0.5)/nb
a.fill_between(xc, D1, color=BLUE, alpha=0.2, lw=0, label="histogram, 1.5×10⁸ iterates")
a.plot([0, 1/8, 1/8, 29/40, 29/40, 1], [10750/3421]*2 + [2000/3421]*2 + [290/311]*2, color=INK, lw=1.6, label="exact invariant density")
a.legend(frameon=False, fontsize=9.5, loc="upper right")
a.set_xlim(0, 1); a.set_ylim(0, 3.6); a.set_xlabel("x"); a.set_ylabel("density")
a.set_title("(b) Invariant density of the uncoupled map f", loc="left", fontsize=12); clean(a)
a = fig.add_subplot(gs[:, 1])
im = a.imshow(D.T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=np.nanquantile(D, 0.995), interpolation="nearest")
a.add_patch(MPoly(S, closed=True, fill=False, ec="white", lw=1.2, ls=(0, (3, 2))))
cS = S.mean(axis=0); a.text(cS[0] + 0.004, cS[1] + 0.004, "S", color="white", fontsize=14, fontweight="bold", ha="center", va="center")
a.set_xlim(0.062, 0.146); a.set_ylim(0.652, 0.847); a.set_aspect("equal"); a.set_xticks([0.07, 0.10, 0.13])
a.set_xlabel("x₁ = x₂  (block)"); a.set_ylabel("x₃  (singleton)")
a.set_title("(c) Invariant density on the attractor", loc="left", fontsize=12)
for sp in a.spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=a, fraction=0.06, pad=0.03); cb.outline.set_edgecolor(AXIS); cb.ax.tick_params(colors=MUTED)
fig.savefig("fig_main.pdf", bbox_inches="tight", dpi=300); fig.savefig("fig_main.png", bbox_inches="tight", dpi=150); plt.close(fig)
print("max |histogram - exact| for the density of f:", np.abs(D1 - np.where(xc < 1/8, 10750/3421, np.where(xc < 29/40, 2000/3421, 290/311))).max().round(4))
