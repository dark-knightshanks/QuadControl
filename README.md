# QuadControl

**MPPI-guided diffusion trajectory learning for stable quadrupedal locomotion on MuJoCo Go2.**

QuadControl is a Julia-based simulation and control pipeline for the Unitree Go2 quadruped robot. It combines **Model Predictive Path Integral (MPPI)** optimal control with **DAgger-refined imitation learning** to distill expensive online trajectory optimization into fast, standalone neural network policies — comparing a standard **MLP Behavior Cloning** baseline against a generative **Trajectory Diffusion Policy**.

---

## Key Features

- **MuJoCo Simulation**: Full Unitree Go2 dynamics via `MuJoCo.jl` with realistic joint limits, torque limits, and PD position control.
- **MPPI Expert Controller**: Parallel stochastic trajectory rollouts ($K=150$, horizon $H=40$) with importance-weighted trajectory updates.
- **Trajectory Diffusion Policy**: $x_0$-prediction diffusion model with sinusoidal time embeddings and accelerated **DDIM sampling** (20 denoising steps) for real-time joint trajectory generation.
- **MLP Behavior Cloning Baseline**: Standard feedforward MLP policy for direct comparison.
- **Non-Invasive DAgger**: Dataset aggregation where the learned policy drives the robot while MPPI relabels states on **cloned simulation snapshots** — preventing simulation pollution during expert queries.
- **4-Frame State History Conditioning**: Both policies receive a sliding window of recent states $[s_{t-3}, s_{t-2}, s_{t-1}, s_t]$ for temporal context (joint accelerations, body pitch rates).

---

## Multi-Phase Training Pipeline

| Phase | Description |
| :--- | :--- |
| **Phase 1–2: MPPI Expert Data Collection** | MPPI controls the quadruped for 500 iterations, recording history-conditioned state → trajectory pairs. |
| **Phase 3: DAgger Correction Rounds** | The learned policy (Diffusion or MLP) drives the robot. MPPI queries on cloned state snapshots relabel actions and aggregate into the training set. Policy is retrained after each round. |
| **Phase 4: Pure Policy Evaluation** | Autonomous control driven entirely by the trained policy — no MPPI. Evaluated over 1,000+ steps. |

---

## Results Summary

| Metric | MLP Behavior Cloning | Diffusion Policy |
| :--- | :--- | :--- |
| **Phase 4 Stability** | Falls at ~Step 470 | **1,000+ steps, zero falls** |
| **Height Tracking** | Sags from 0.29m → 0.22m | Steady 0.30–0.31m |
| **Forward Velocity** | Overspeeds to 0.51 m/s → collapse | Smooth 0.15–0.34 m/s |
| **Failure Mode** | Covariate shift, mode-averaging | None observed |

The Diffusion Policy's iterative denoising process preserves sharp trajectory modes rather than averaging them, making it robust to out-of-distribution states where the standard MLP breaks down.

---

## Repository Structure

```
QuadControl/
├── go2/                          # Unitree Go2 robot assets
│   ├── scene.xml                 #   Flat ground MuJoCo scene
│   ├── scene_terrain.xml         #   Terrain/obstacle scene
│   ├── go2.xml                   #   Robot MJCF definition
│   └── assets/                   #   Meshes and textures
├── models/
│   ├── mppi-pd.jl                # Main pipeline: MPPI + Diffusion Policy + DAgger
│   ├── mppi-bc.jl                # Comparison pipeline: MPPI + MLP Behavior Cloning + DAgger
│   ├── diffusion1_model.jl       # Diffusion model architecture, training, and DDIM sampler
│   ├── mlp_policy.jl             # MLP policy architecture and BC training
│   └── plot_metrics.py           # Metrics visualization script
├── metrics.csv                   # Recorded MPPI cost/state metrics
├── metrics_plot.png              # Visualization of training metrics
├── Project.toml                  # Julia project dependencies
├── Manifest.toml                 # Julia dependency lock file
└── README.md
```

---

## Getting Started

### Prerequisites

- [Julia](https://julialang.org/) (v1.8+)
- Julia packages: `MuJoCo.jl`, `Flux.jl`, `LinearAlgebra`, `Statistics`, `Random`

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dark-knightshanks/QuadControl.git
   cd QuadControl
   ```

2. **Instantiate Julia dependencies**:
   ```julia
   using Pkg
   Pkg.instantiate()
   ```

### Running the Pipelines

**Diffusion Policy pipeline** (MPPI → DAgger → Diffusion):
```bash
julia models/mppi-pd.jl
```

**Behavior Cloning pipeline** (MPPI → DAgger → MLP):
```bash
julia models/mppi-bc.jl
```
