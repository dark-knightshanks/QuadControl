"""
Visualize MPPI-PD training metrics from metrics.csv
Run: python3 models/plot_metrics.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── Load Data ─────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "metrics.csv"
if not csv_path.exists():
    csv_path = Path("metrics.csv")   # fallback: cwd

data = np.genfromtxt(csv_path, delimiter=",", names=True)

iters = data["iter"]

# Check if sym_cost column exists (new structural cost function)
has_sym = "sym_cost" in data.dtype.names

# ── Colour palette ────────────────────────────────────────────────────
C = {
    "cost":    "#e63946",
    "height":  "#457b9d",
    "vel":     "#2a9d8f",
    "quat":    "#e9c46a",
    "joint":   "#f4a261",
    "sym":     "#b5838d",
    "lateral": "#264653",
    "jvel":    "#6a4c93",
    "sat":     "#d62828",
    "torque":  "#f77f00",
    "target":  "#adb5bd",
}

fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=120)
fig.patch.set_facecolor("#0f1117")
for ax in axes.flat:
    ax.set_facecolor("#181c24")
    ax.tick_params(colors="#c9d1d9", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, color="#30363d", linewidth=0.4, alpha=0.6)
    ax.set_xlabel("Iteration", color="#8b949e", fontsize=9)

# ── 1. Total min cost ─────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(iters, data["min_cost"], color=C["cost"], linewidth=1.2, alpha=0.85)
ax.fill_between(iters, data["min_cost"], alpha=0.08, color=C["cost"])
ax.set_title("Min Rollout Cost", color="#c9d1d9", fontsize=11, fontweight="bold")
ax.set_ylabel("Cost", color="#8b949e", fontsize=9)

# ── 2. Cost breakdown (stacked area) ─────────────────────────────────
ax = axes[0, 1]
components = [
    ("height_cost",  "Height",   C["height"]),
    ("quat_cost",    "Quat",     C["quat"]),
    ("joint_cost",   "Jnt Range",C["joint"]),
]
if has_sym:
    components.append(("sym_cost", "Symmetry", C["sym"]))
components += [
    ("vel_cost",     "Velocity", C["vel"]),
    ("lateral_cost", "Lateral",  C["lateral"]),
    ("jvel_cost",    "Jnt Vel",  C["jvel"]),
]
bottoms = np.zeros_like(iters, dtype=float)
for key, label, color in components:
    vals = data[key]
    ax.bar(iters, vals, bottom=bottoms, color=color, alpha=0.75, width=1.0, label=label)
    bottoms += vals
ax.legend(fontsize=7, loc="upper right", framealpha=0.3, labelcolor="#c9d1d9",
          facecolor="#181c24", edgecolor="#30363d")
ax.set_title("Cost Breakdown (stacked)", color="#c9d1d9", fontsize=11, fontweight="bold")
ax.set_ylabel("Cost", color="#8b949e", fontsize=9)

# ── 3. Height tracking ───────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(iters, data["height"], color=C["height"], linewidth=1.2, alpha=0.85, label="Actual")
ax.axhline(0.38, color=C["target"], linestyle="--", linewidth=1, alpha=0.6, label="Target (0.445)")
ax.axhline(0.20,  color=C["sat"], linestyle=":", linewidth=1, alpha=0.5, label="Fall threshold")
ax.fill_between(iters, data["height"], 0.445, alpha=0.06, color=C["height"])
ax.legend(fontsize=7, loc="lower right", framealpha=0.3, labelcolor="#c9d1d9",
          facecolor="#181c24", edgecolor="#30363d")
ax.set_title("Body Height", color="#c9d1d9", fontsize=11, fontweight="bold")
ax.set_ylabel("Height (m)", color="#8b949e", fontsize=9)

# ── 4. Forward velocity ──────────────────────────────────────────────
ax = axes[1, 1]
ax.plot(iters, data["vel_x"], color=C["vel"], linewidth=1.2, alpha=0.85, label="Actual")
ax.axhline(0.3, color=C["target"], linestyle="--", linewidth=1, alpha=0.6, label="Target (0.3 m/s)")
ax.axhline(0.0, color="#6e7681", linestyle=":", linewidth=0.8, alpha=0.4)
ax.fill_between(iters, data["vel_x"], 0.3, alpha=0.06, color=C["vel"])
ax.legend(fontsize=7, loc="lower right", framealpha=0.3, labelcolor="#c9d1d9",
          facecolor="#181c24", edgecolor="#30363d")
ax.set_title("Forward Velocity", color="#c9d1d9", fontsize=11, fontweight="bold")
ax.set_ylabel("vel_x (m/s)", color="#8b949e", fontsize=9)

# ── 5. Saturation count ──────────────────────────────────────────────
ax = axes[2, 0]
ax.bar(iters, data["n_saturated"], color=C["sat"], alpha=0.7, width=1.0)
ax.set_ylim(0, 12.5)
ax.axhline(6, color=C["target"], linestyle="--", linewidth=0.8, alpha=0.4, label="50 % clipped")
ax.legend(fontsize=7, loc="upper right", framealpha=0.3, labelcolor="#c9d1d9",
          facecolor="#181c24", edgecolor="#30363d")
ax.set_title("Joints Saturated  (/ 12)", color="#c9d1d9", fontsize=11, fontweight="bold")
ax.set_ylabel("Count", color="#8b949e", fontsize=9)

# ── 6. Max raw torque ────────────────────────────────────────────────
ax = axes[2, 1]
ax.plot(iters, data["max_raw_torque"], color=C["torque"], linewidth=1.2, alpha=0.85)
ax.axhline(23.7,  color="#e5e5e5", linestyle="--", linewidth=0.8, alpha=0.35, label="Hip limit (23.7)")
ax.axhline(45.43, color="#e5e5e5", linestyle=":",  linewidth=0.8, alpha=0.35, label="Knee limit (45.43)")
ax.fill_between(iters, data["max_raw_torque"], alpha=0.06, color=C["torque"])
ax.legend(fontsize=7, loc="upper right", framealpha=0.3, labelcolor="#c9d1d9",
          facecolor="#181c24", edgecolor="#30363d")
ax.set_title("Max Raw Torque (pre-clamp)", color="#c9d1d9", fontsize=11, fontweight="bold")
ax.set_ylabel("Torque (N·m)", color="#8b949e", fontsize=9)

# ── Layout & save ────────────────────────────────────────────────────
fig.suptitle("MPPI-PD  ·  300 Iteration Run  (KP=80, KD=4, structural cost)",
             color="#e6edf3", fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = csv_path.parent / "metrics_plot.png"
fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
print(f"✓ Plot saved to {out_path}")
plt.show()
