# Attracting chimeras with chaotic dynamics in globally coupled maps

Code accompanying the paper

> Théophile Caby, *Examples of attracting chimeras with chaotic dynamics in globally coupled maps* (2026).

It contains the computer-assisted parts of the proofs, in exact rational arithmetic, and all the numerical
computations and figures of the paper.

The system is the globally coupled map (GCM)

$$x_i^{t+1} = (1-\varepsilon)\, f(x_i^t) + \frac{\varepsilon}{N}\sum_{j=1}^N f(x_j^t), \qquad i = 1,\dots,N,$$

with $\varepsilon = 1/4$ and piecewise linear local maps $f$. A chimera is an attractor on which some units are
synchronized while the others are not.

## Quick start

The exact verifications need only Python 3 (standard library) and take about 30 seconds:

```
python verify_all.py
```

Each check prints `OK` or `FAIL`, followed by `ALL CHECKS PASSED` for each example.

The numerical scripts need `numpy`, `numba`, `scipy`, `matplotlib` and `mpmath`:

```
pip install -r requirements.txt
```

Run each script from its own folder (for example `cd three_units && python simulations.py`). Most numerical
scripts accept `--quick` for a smaller, faster run.

## Contents

The examples are organized by folder. Section and statement numbers refer to the paper.

### `three_units/`: the main example (N = 3, six-piece ergodic local map f)

| Script | Paper | What it does |
|---|---|---|
| `verify_exact.py` | Prop. 3.1, Thm. 3.2, Prop. 3.3, Prop. 3.4; App. A.2–A.8 | Exact check of every number in these statements and proofs: Markov structure and invariant density of f, trapping region and margin 1/48, invariant parallelogram S, uniform expansion, and the exact attractor G⁷(R) = G⁶(R) (about 20 s). |
| `exact_polygons.py` | App. A.7 | Library: exact images and areas of unions of convex rational polygons. |
| `exact_attractor.py` | Prop. 3.3 | Computes G<sup>n</sup>(R) exactly and saves the polygons to `exact_levels.pkl` (used by `occupancy.py`). |
| `robustness.py` | Prop. 3.5(i); App. A.9 | Numbers used in the robustness proof: margins 1/128 and 1/480, d > 3.9·10⁻⁴, operator norms, smoothing bound. |
| `common.py` |  | The map f, its derivative, the reduced map G and the full GCM (numba). |
| `simulations.py` | Section 5 | Lyapunov exponents (≈ 0.41, 0.12) and TLE on the chimera; fate of 10⁶ random initial conditions (a third to each chimera); time to enter a trapping box (median 71 steps). |
| `birkhoff.py` | Section 5 | Birkhoff averages of the TLE (log 15/16) and of the averaged constraint of Prop. 4.3 (0). |
| `occupancy.py` | Section 5 | Long orbits visit every cell of a fine grid whose center lies in the exact attractor. |
| `density_hist.py` | Section 5, Fig. 1(c) | Empirical invariant density on the attractor (10⁹ points), saved to `density_hist.npz`. |
| `density_check.py` | Section 5 | Boundedness of the invariant density, evaluated exactly through backward branches (no binning). |
| `make_fig_main.py` | Figure 1 | Produces `fig_main.pdf` (run `density_hist.py` first). |

### `four_units/`: two synchronized and two unsynchronized units (N = 4)

| Script | Paper | What it does |
|---|---|---|
| `verify_exact.py` | Thm. 3.6; App. A.10 | Exact check: Markov structure and density of f̂, trapping region, uniform expansion (leading minors of M̂ᵀM̂ − I), the ball of radius 34/1473 in the attractor, the TLE log(5/4) of the free pair. |
| `simulations.py` | Section 3.1 | Basins from random initial conditions in [0,1]⁴, Lyapunov exponents (≈ 0.30, 0.30, 0.07), distance between the free units. |
| `make_fig_twofree.py` | Figure 2 | Produces `fig_twofree.pdf`. |

### `four_piece/`: the non-ergodic four-piece example (N = 3)

| Script | Paper | What it does |
|---|---|---|
| `verify_exact.py` | Thm. 6.1; App. A.12 | Exact check of the data of the proof, including the exact attractor G̃⁴(R̃) = G̃³(R̃). |
| `exact_polygons.py` | Thm. 6.1 | Polygon library for f̃. |
| `basins.py`, `final_states.py` | Section 6.2 | Fate of random initial conditions (about a quarter go to the chimeras). |

### `logistic_henon/`: the Hénon-type chimeras of the logistic map (Appendix B)

| Script | What it does |
|---|---|
| `gcm_logistic.py` | Reduced map, TLE and Lyapunov exponents for f(x) = 4x(1−x), N = 3. |
| `fine_scan.py` | Scan of ε around 0.302–0.304: TLE and Lyapunov exponents. |
| `chaos_tests.py` | Largest exponent under random perturbations, and over blocks of 10⁷ steps along orbits of 10⁹ steps. |
| `highprec.py` | Comparison with 60-digit arithmetic. |
| `po_fast.py`, `po_check.py` | Periodic orbits on the attractor: stability and transverse exponents. |
| `bursts.py` | Full system from 300 random initial conditions: no desynchronization bursts over 2·10⁶ steps. |

### `search/`: how the examples were found

Exploratory scripts kept for transparency; they are not needed to check the results.
`search/three_units/` contains the search over six-piece Markov maps that produced f (`search.py`, then
`merge_test.py`). `search/four_units/` contains a linear-programming exploration (`lp_search.py`) and the
template search that produced f̂ (`ergodic_search.py`, `candidates.py`).

## Run times

Measured on a 2-core virtual machine. The exact verifications take about 30 s in total. Full numerical runs take
from a few seconds to a few minutes; `--quick` runs take under a minute each. Some scripts in `logistic_henon/`
(`chaos_tests.py`, `bursts.py`, `po_check.py`) take 10–20 minutes. The first run of a numba
script also includes compilation.

## Requirements

Python ≥ 3.9. Tested with Python 3.13, numpy 2.5, numba 0.68, scipy 1.18, matplotlib 3.11 and mpmath 1.3.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## Funding

This work was funded by the European Space Agency through the GRID4EARTH project (ESA contract 4000147951/25/I-NS).

## Use of AI

As stated in the paper, the code in this repository was written by Claude Opus 5.5 (Anthropic), an AI system,
working under the supervision of the author, who takes full responsibility for it.

## License

MIT (see `LICENSE`).
