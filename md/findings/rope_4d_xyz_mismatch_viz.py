"""
Visualizations for md/findings/rope_4d_xyz_mismatch.md
- Panel A: inv_freq / wavelength spectrum with useful-Δp window
- Panel B: usable-band heatmap over (k, Δp) with Qwen-text / 4D-xyz overlay
- Panel C: Sequential vs Interleaved band-per-axis utilization diagram
- Panel D: Bar chart of useful-band count per axis
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
OUT_DIR = Path(__file__).parent
OUT_PNG = OUT_DIR / "rope_4d_xyz_mismatch_viz.png"

# ---- RoPE constants (Qwen3.5-4B) ----
THETA = 1e7
D_ROT = 64
N_BANDS = D_ROT // 2  # 32
K = np.arange(N_BANDS)
INV_FREQ = THETA ** (-K / N_BANDS)           # = 1e7 ** (-k/32)
WAVELEN = 2 * math.pi / INV_FREQ

LOWER = 0.01           # rad, below: band is DC
UPPER = 2 * math.pi    # rad, above: phase wraps

# --- Δp envelopes ---
# Qwen text: positions 1..128K
QWEN_DP_MIN, QWEN_DP_MAX = 1, 128_000
# 4D xyz (coord_scale=100, indoor scene): 10..2000
XYZ_DP_MIN, XYZ_DP_MAX = 10, 2_000

# --- mrope_section [2,10,10,10] -> per-axis band slices ---
AXIS_SLICES = {
    "t": (0, 2),
    "x": (2, 12),
    "y": (12, 22),
    "z": (22, 32),
}
# per-axis Δp range (position units, coord_scale=100)
AXIS_DP_MIN = {"t": 1, "x": 10, "y": 10, "z": 10}
AXIS_DP_MAX = {"t": 50, "x": 2000, "y": 2000, "z": 2000}
AXIS_COLORS = {"t": "#4C72B0", "x": "#55A868", "y": "#CCB974", "z": "#C44E52"}


def band_useful(inv_f: float, dp_min: float, dp_max: float) -> bool:
    """Band k is useful for axis if its useful-window [LOWER/inv_f, UPPER/inv_f]
    overlaps the axis's Δp range [dp_min, dp_max]."""
    return (dp_max * inv_f >= LOWER) and (dp_min * inv_f <= UPPER)


# ==================================================================
fig = plt.figure(figsize=(20, 13))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                      left=0.06, right=0.97, top=0.92, bottom=0.07)

# --------- Panel A: spectrum + useful window ---------
axA = fig.add_subplot(gs[0, 0])
axA.semilogy(K, INV_FREQ, "o-", color="#222", label="inv_freq[k]")
axA.set_xlabel("band index k")
axA.set_ylabel("inv_freq[k]   (log)")
axA.set_title("A. RoPE frequency spectrum  (Qwen3.5,  θ=1e7, d_rot=64)")
axA.grid(True, which="both", alpha=0.3)

# Dual y-axis: wavelength
axA2 = axA.twinx()
axA2.semilogy(K, WAVELEN, color="#999", ls="--", lw=1)
axA2.set_ylabel("wavelength = 2π / inv_freq   (log)", color="#666")

# Shade "useful for Qwen text" (Δp up to 128K)
#   usable band has inv_freq in [LOWER/Δp_max, UPPER/Δp_min]
def useful_range(dp_min, dp_max):
    # inv_freq window covering all |Δp| in [dp_min,dp_max]
    lo = LOWER / dp_max     # highest-k (slowest) bound
    hi = UPPER / dp_min     # lowest-k (fastest) bound
    return lo, hi

qwen_lo, qwen_hi = useful_range(QWEN_DP_MIN, QWEN_DP_MAX)
xyz_lo,  xyz_hi  = useful_range(XYZ_DP_MIN,  XYZ_DP_MAX)

axA.axhspan(qwen_lo, qwen_hi, color="#4C72B0", alpha=0.10,
            label=f"useful for Qwen text (Δp∈[1,128K])")
axA.axhspan(xyz_lo,  xyz_hi,  color="#C44E52", alpha=0.18,
            label=f"useful for 4D xyz  (Δp∈[10,2K], scale=100)")

# Mark the dead region for xyz
dead_mask = INV_FREQ < xyz_lo
dead_ks = K[dead_mask]
if len(dead_ks):
    axA.axvspan(dead_ks.min() - 0.5, dead_ks.max() + 0.5,
                color="#C44E52", alpha=0.08, lw=0)
    axA.text(dead_ks.mean(), INV_FREQ.min() * 1.5,
             f"dead for xyz\n(k={dead_ks.min()}–{dead_ks.max()})",
             ha="center", va="bottom", color="#8B0000", fontsize=9)

axA.legend(loc="upper right", fontsize=9)

# --------- Panel B: usability heatmap over (k, Δp) ---------
axB = fig.add_subplot(gs[0, 1])
DP_GRID = np.logspace(0, 6, 240)     # 1 .. 1e6
PHI = np.outer(DP_GRID, INV_FREQ)     # shape (Ndp, Nk)
state = np.zeros_like(PHI)            # 0 = DC(dead), 1 = usable, 2 = aliased
state[PHI >= LOWER] = 1
state[PHI > UPPER] = 2

cmap = plt.matplotlib.colors.ListedColormap(
    ["#B00020", "#2E7D32", "#F9A825"])
axB.imshow(state, origin="lower", aspect="auto",
           cmap=cmap, vmin=0, vmax=2,
           extent=[-0.5, N_BANDS - 0.5,
                   math.log10(DP_GRID[0]), math.log10(DP_GRID[-1])])
axB.set_xlabel("band index k")
axB.set_ylabel("log10(Δp)")
axB.set_title("B. Band usability vs (k, Δp)")

# Overlay: Qwen text & 4D xyz envelopes
for (dp_min, dp_max, color, label) in [
    (QWEN_DP_MIN, QWEN_DP_MAX, "#4C72B0", "Qwen text  (1 → 128K)"),
    (XYZ_DP_MIN, XYZ_DP_MAX, "#FFFFFF", "4D xyz  (10 → 2K, scale=100)"),
]:
    axB.axhspan(math.log10(dp_min), math.log10(dp_max),
                xmin=0.0, xmax=1.0, fill=False,
                edgecolor=color, lw=2, label=label)

legend_items = [
    mpatches.Patch(color="#B00020", label="dead  (|Δφ|<0.01)"),
    mpatches.Patch(color="#2E7D32", label="usable  (0.01 ≤ |Δφ| ≤ 2π)"),
    mpatches.Patch(color="#F9A825", label="aliased  (|Δφ|>2π)"),
    plt.Line2D([], [], color="#4C72B0", lw=2, label="Qwen text Δp window"),
    plt.Line2D([], [], color="#FFFFFF", lw=2, label="4D xyz Δp window"),
]
axB.legend(handles=legend_items, loc="lower right",
           fontsize=8, framealpha=0.85)

# --------- Panel C: Sequential [2,10,10,10] visualization ---------
axC = fig.add_subplot(gs[1, 0])
axC.set_xlim(-0.5, N_BANDS + 7)
axC.set_ylim(-0.5, 4.5)
axC.set_yticks(range(4))
axC.set_yticklabels(["z", "y", "x", "t"])      # rows top→bottom: t x y z
axC.invert_yaxis()
axC.set_xlabel("band index k")
axC.set_title("C. Sequential mrope_section = [2,10,10,10]:  which bands each axis can actually use")

for row, ax_name in enumerate(["t", "x", "y", "z"]):
    lo, hi = AXIS_SLICES[ax_name]
    dp_min, dp_max = AXIS_DP_MIN[ax_name], AXIS_DP_MAX[ax_name]
    # thin base bar for the slice
    axC.add_patch(plt.Rectangle((lo - 0.45, row - 0.38), hi - lo - 0.10, 0.76,
                                facecolor=AXIS_COLORS[ax_name], alpha=0.18,
                                edgecolor="none"))
    # usable vs dead within the slice
    for k in range(lo, hi):
        usable = band_useful(INV_FREQ[k], dp_min, dp_max)
        color = AXIS_COLORS[ax_name] if usable else "#D0D0D0"
        edge = "#222"
        axC.add_patch(plt.Rectangle((k - 0.42, row - 0.34), 0.84, 0.68,
                                    facecolor=color, edgecolor=edge, lw=0.6))
        if not usable:
            axC.text(k, row, "×", ha="center", va="center",
                     color="#A00000", fontsize=11, fontweight="bold")

    used = sum(band_useful(INV_FREQ[k], dp_min, dp_max)
               for k in range(lo, hi))
    axC.text(N_BANDS + 0.2, row,
             f"{used}/{hi - lo} useful    (Δp ∈ [{dp_min},{dp_max}])",
             va="center", fontsize=10,
             color=AXIS_COLORS[ax_name], fontweight="bold")

axC.set_xticks(range(0, N_BANDS, 2))
axC.set_xticklabels([str(k) for k in range(0, N_BANDS, 2)])
axC.grid(False)
axC.spines["top"].set_visible(False)
axC.spines["right"].set_visible(False)
# hide tick-labels in the right "padding" region (>=N_BANDS)
axC.tick_params(axis="x", which="both", labelsize=9)

# --------- Panel D: Interleaved vs Sequential useful-band bar chart ---------
axD = fig.add_subplot(gs[1, 1])

# Sequential counts
seq_useful = {}
for ax_name, (lo, hi) in AXIS_SLICES.items():
    dp_min, dp_max = AXIS_DP_MIN[ax_name], AXIS_DP_MAX[ax_name]
    seq_useful[ax_name] = sum(
        band_useful(INV_FREQ[k], dp_min, dp_max) for k in range(lo, hi))

# Interleaved: bands 2..31 round-robin across x,y,z (each gets 10 bands);
# t stays at 0,1.  Axis's bands are a uniform sample of the 30 spatial bands.
interleave_bands = {"t": [0, 1], "x": [], "y": [], "z": []}
axis_order = ["x", "y", "z"]
for i, k in enumerate(range(2, N_BANDS)):
    interleave_bands[axis_order[i % 3]].append(k)

inter_useful = {}
for ax_name, bands in interleave_bands.items():
    dp_min, dp_max = AXIS_DP_MIN[ax_name], AXIS_DP_MAX[ax_name]
    inter_useful[ax_name] = sum(
        band_useful(INV_FREQ[k], dp_min, dp_max) for k in bands)

axes_order = ["t", "x", "y", "z"]
xs = np.arange(len(axes_order))
w = 0.36
bars1 = axD.bar(xs - w / 2, [seq_useful[a] for a in axes_order],
                width=w, label="Sequential  [2,10,10,10]",
                color=[AXIS_COLORS[a] for a in axes_order],
                edgecolor="#333")
bars2 = axD.bar(xs + w / 2, [inter_useful[a] for a in axes_order],
                width=w, label="Interleaved vision",
                color=[AXIS_COLORS[a] for a in axes_order],
                edgecolor="#333", hatch="//", alpha=0.75)

# Annotate bar heights
for bars in (bars1, bars2):
    for b in bars:
        h = b.get_height()
        axD.text(b.get_x() + b.get_width() / 2, h + 0.15, f"{int(h)}",
                 ha="center", va="bottom", fontsize=9)

axD.set_xticks(xs)
axD.set_xticklabels(axes_order)
axD.set_ylabel("# useful bands (of 10 allocated per xyz axis; 2 for t)")
axD.set_title("D. Useful-band count per axis  (4D xyz,  max Δp from Panel C)")
axD.set_ylim(0, max(max(seq_useful.values()), max(inter_useful.values())) + 1.5)
axD.grid(axis="y", alpha=0.3)
axD.legend(loc="upper right", fontsize=9)

# Footer note
fig.suptitle(
    "4D M-RoPE for continuous xyz — frequency-band mismatch\n"
    "Qwen's ladder was tuned for 5-order text Δp; physical coords have only 2 orders",
    fontsize=14, y=0.985,
)

fig.savefig(OUT_PNG, dpi=140)
print(f"Wrote {OUT_PNG}")
print()
print("Useful-band counts under sequential [2,10,10,10]:")
for a in axes_order:
    lo, hi = AXIS_SLICES[a]
    print(f"  {a}: {seq_useful[a]} / {hi - lo}  (max Δp={AXIS_DP_MAX[a]})")
print()
print("Useful-band counts under interleaved vision:")
for a in axes_order:
    n_total = len(interleave_bands[a])
    print(f"  {a}: {inter_useful[a]} / {n_total}  (bands = {interleave_bands[a]})")
