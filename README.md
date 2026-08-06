# QuadControl

**QuadControl** is a Julia-based simulation environment and control pipeline for quadruped height regulation and motion control on the Unitree Go2 robot using MuJoCo.

The project combines **Model Predictive Path Integral (MPPI)** control with **Data Aggregation (DAgger)** to distill a high-performance, sampling-based optimal controller into a lightweight neural network (MLP).

---

## Key Features

- **MuJoCo Simulation Integration**: Leverages `MuJoCo.jl` to simulate the Unitree Go2 quadruped dynamics accurately with joint limits and realistic torque limits.
- **MPPI Controller**: Real-time sampling-based optimal control using parallel trajectory rollouts ($K=150$ rollouts over a horizon $H=40$).
- **DAgger & Behavior Cloning**: Distills the computationally expensive MPPI controller into a fast Multilayer Perceptron (MLP) policy.
- **Multi-Phase Training Pipeline**:
  1. **Phase 1–2 (MPPI Expert)**: Expert MPPI controls the quadruped while recording state-action trajectory data.
  2. **Phase 3 (DAgger Phase)**: Neural network policy controls the robot while MPPI provides continuous corrections and data aggregation.
  3. **Phase 4 (Pure MLP)**: Autonomous control driven strictly by the trained neural network without online MPPI optimization.

---

## Repository Structure

```
QuadControl/
├── go2/                  # Go2 robot assets and MuJoCo scene definition (scene.xml)
├── models/               # Saved model checkpoints / neural network weights
├── mlp_policy.jl         # Neural network policy architecture and training utilities (BC/DAgger)
├── metrics.csv           # Recorded performance metrics across phases
├── metrics_plot.png      # Visualization plot for metrics (height tracking, rollouts, losses)
└── README.md             # Project overview and documentation
```

---

## Getting Started

### Prerequisites

- [Julia](https://julialang.org/) (v1.8+ recommended)
- `MuJoCo.jl` and Julia packages: `LinearAlgebra`, `Random`, `Statistics`

### Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/QuadControl.git
   cd QuadControl
   ```

2. **Instantiate Julia Dependencies**:
   ```julia
   using Pkg
   Pkg.instantiate()
   ```
