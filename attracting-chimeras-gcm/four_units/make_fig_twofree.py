"""
Figure 2 of the paper (fig_twofree.pdf): the four-unit example. (a) the local map f^, (b) a time series from a point
of U^, (c) the empirical joint density of the two free units on the attractor. Needs numpy and matplotlib; about a minute.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
e = 0.25
XS = np.array([0, 1/30, 1/6, 3/10, 4/5, 23/25, 1.0]); YS = np.array([1, 1/6, 0, 1/6, 1, 4/5, 0])
f = lambda x: np.interp(x, XS, YS)
def F(x):
    fx = f(x); return (1-e)*fx + e*fx.mean(axis=-1, keepdims=True)
P = (1/30, 3/10); Q = (17/25, 23/25)
fig = plt.figure(figsize=(16, 4.8))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.35, 1.05], wspace=0.28, hspace=0.12)
axs = [fig.add_subplot(gs[:, 0]), None, fig.add_subplot(gs[:, 2])]
bt, bb = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
# (a) local map
a = axs[0]
a.axvspan(*P, color=BLUE, alpha=0.12, lw=0); a.axvspan(*Q, color=ORANGE, alpha=0.12, lw=0)
a.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=(0, (4, 3))); a.plot(XS, YS, color=INK, lw=1.7, marker="o", ms=3.5)
a.text(np.mean(P), 0.62, r"$\hat P$" + "\npair\n" + r"$|\hat f\,'| = 5/4$", ha="center", va="top", fontsize=9, color=INK2)
a.text(np.mean(Q), 0.42, r"$\hat Q$" + "\ntwo free\nunits\n" + r"$|\hat f\,'| = 5/3$", ha="center", va="top", fontsize=9, color=INK2)
a.set_xlim(0, 1); a.set_ylim(0, 1.02); a.set_xlabel("x")
a.set_title(r"(a) Local map $\hat f$: 6 linear pieces", loc="left", fontsize=12); clean(a)
# (b) time series from a point of U, zoomed on Q (top) and P (bottom)
x = np.array([0.06, 0.28, 0.72, 0.87]); T = 60; ts = [x]
for _ in range(T): x = F(x); ts.append(x)
ts = np.array(ts); t = np.arange(T+1)
a = bt
a.plot(t, ts[:, 2], color=AQUA, lw=1.3, marker="o", ms=2.8, label="x₃")
a.plot(t, ts[:, 3], color=PURPLE, lw=1.3, marker="o", ms=2.8, label="x₄")
a.set_xlim(0, T); a.set_ylim(0.70, 0.90); a.tick_params(labelbottom=False)
a.legend(frameon=False, loc="lower right", ncol=2, fontsize=9); a.set_ylabel(r"in $\hat Q$")
a.set_title("(b) x₃ and x₄ stay apart; x₁ and x₂ merge", loc="left", fontsize=12); clean(a)
a = bb
a.plot(t, ts[:, 0], color=BLUE, lw=2.0, marker="o", ms=3.2, label="x₁")
a.plot(t, ts[:, 1], color=ORANGE, lw=1.2, marker="o", ms=2.2, label="x₂")
a.set_xlim(0, T); a.set_ylim(0.04, 0.30); a.set_xlabel("t"); a.set_ylabel(r"in $\hat P$")
a.legend(frameon=False, loc="upper right", ncol=2, fontsize=9); clean(a)
# (c) density of the two free units on the attractor
rng = np.random.default_rng(3); K = 4000
X = np.column_stack([rng.uniform(*P, K), rng.uniform(*P, K), rng.uniform(*Q, K), rng.uniform(*Q, K)])
for _ in range(400): X = F(X)
nb = 260; lo, hi = 0.74, 0.90; H = np.zeros((nb, nb))
for _ in range(2500):
    X = F(X)
    i = np.clip(((X[:, 2]-lo)/(hi-lo)*nb).astype(int), 0, nb-1); j = np.clip(((X[:, 3]-lo)/(hi-lo)*nb).astype(int), 0, nb-1)
    np.add.at(H, (i, j), 1)
H = H/H.sum()/((hi-lo)/nb)**2; H[H == 0] = np.nan
a = axs[2]
im = a.imshow(H.T, origin="lower", extent=[lo, hi, lo, hi], cmap="viridis", vmin=0, vmax=np.nanquantile(H, 0.995), interpolation="nearest")
a.plot([lo, hi], [lo, hi], color="white", lw=1.0, ls=(0, (4, 3)))
a.text(0.765, 0.785, "x₃ = x₄", color="white", fontsize=9, ha="left", va="bottom", rotation=45)
a.set_xlabel("x₃"); a.set_ylabel("x₄")
cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03); cb.outline.set_visible(False); cb.ax.tick_params(colors=MUTED, labelsize=8)
a.set_title("(c) Joint density of the free units", loc="left", fontsize=12)
fig.savefig("fig_twofree.pdf", bbox_inches="tight", dpi=300); fig.savefig("fig_twofree.png", bbox_inches="tight", dpi=110)
print("saved; free-unit ranges", np.nanmin(X[:,2:]), np.nanmax(X[:,2:]))
