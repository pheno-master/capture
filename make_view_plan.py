"""Regenerate figures/view_plan.pdf from the capture view planner.

Usage: python3 make_view_plan.py <path to capture repo> <scan config .yaml> [out stem]
"""
import sys
from pathlib import Path

repo, config = Path(sys.argv[1]), Path(sys.argv[2])
out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).with_name("view_plan")
sys.path.insert(0, str(repo))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.core.config import load_config
from src.core.views import hemisphere_views, ring_poses

cfg = load_config(config)
spec = cfg.views
c = np.array(spec.center)
views = hemisphere_views(spec)
order = np.array([T[:3, 3] for T in views])
rings = [np.array([T[:3, 3] for T in ring_poses(spec, el, n)]) for el, n in zip(spec.elevations_deg, spec.ring_sizes())]
colors = ["#2a78d6", "#eb6834", "#1baf7a"]
markers = ["o", "s", "^"]
ink, muted, grid = "#2b2b2a", "#6b6a64", "#d9d8d2"

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": muted,
                     "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted})
fig = plt.figure(figsize=(6.3, 2.9))

# --- left: 3D view -----------------------------------------------------------
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot(*(order - c).T * 100, color=grid, lw=1.0, zorder=1)
for i, (pts, col, mk, el) in enumerate(zip(rings, colors, markers, spec.elevations_deg)):
    p = (pts - c) * 100
    ax.scatter(*p.T, color=col, marker=mk, s=22, depthshade=False, edgecolors="white", linewidths=0.6,
               label=f"{el:.0f}° ({len(pts)} views)")
    for q in pts:  # short viewing-direction ticks
        d = (c - q) / np.linalg.norm(c - q) * 0.035
        ax.plot(*np.stack([q - c, q - c + d]).T * 100, color=col, lw=1.0)
ax.scatter([0], [0], [0], color=ink, marker="*", s=60)
ax.set_xlabel("x [cm]", labelpad=-6); ax.set_ylabel("y [cm]", labelpad=-6); ax.set_zlabel("z [cm]", labelpad=-8)
ax.tick_params(pad=-3, labelsize=7)
ax.view_init(elev=22, azim=-60)
ax.set_box_aspect((1, 1, 0.75))
for a in (ax.xaxis, ax.yaxis, ax.zaxis):
    a.pane.set_facecolor("white"); a.pane.set_edgecolor(grid); a._axinfo["grid"]["color"] = grid
ax.set_title("(a) Camera positions", fontsize=9, color=ink)

# --- right: top view with capture order -------------------------------------
bx = fig.add_subplot(1, 2, 2)
bx.plot(*(order[:, :2] - c[:2]).T * 100, color=grid, lw=1.0, zorder=1)
for pts, col, mk, el in zip(rings, colors, markers, spec.elevations_deg):
    p = (pts[:, :2] - c[:2]) * 100
    bx.scatter(*p.T, color=col, marker=mk, s=26, edgecolors="white", linewidths=0.6, zorder=3,
               label=f"{el:.0f}° ({len(pts)} views)")
    r = np.linalg.norm(p[0]); a = np.radians(180 / len(pts) + 90 + 360 / len(pts) * (len(pts) // 4))
    bx.text(r * np.cos(a), r * np.sin(a), f"{el:.0f}°", color=ink, fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
s, e = (order[0, :2] - c[:2]) * 100, (order[-1, :2] - c[:2]) * 100
bx.annotate("start", s, xytext=(s[0] + 3, s[1] - 4), fontsize=8, color=ink, arrowprops=dict(arrowstyle="-", color=muted, lw=0.6))
bx.annotate("end", e, xytext=(e[0] + 3, e[1] + 3), fontsize=8, color=ink, arrowprops=dict(arrowstyle="-", color=muted, lw=0.6))
bx.scatter([0], [0], color=ink, marker="*", s=60, zorder=4, label="plant centre")
bx.set_aspect("equal"); bx.set_xlabel("x [cm]"); bx.set_ylabel("y [cm]")
bx.grid(color=grid, lw=0.5); bx.set_axisbelow(True)
for sp in ("top", "right"): bx.spines[sp].set_visible(False)
bx.set_title("(b) Top view and capture order", fontsize=9, color=ink)
h, l = bx.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8, title="Ring elevation", title_fontsize=8,
           bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.2, wspace=0.25)
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=130)
print(len(views), "views")
