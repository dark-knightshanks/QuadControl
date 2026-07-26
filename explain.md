# Go2 MPPI + DAgger Controller — Explained

## The big picture first

Before the line-by-line: this file does **three things in sequence**, on one running robot:

1. **Phase 1–2 (MPPI expert only):** Use MPPI (a sampling-based optimal controller) to make the robot stand/walk well, and record `(state, action)` pairs as training data.
2. **Phase 3 (DAgger):** A neural net policy now *drives* the robot, but MPPI keeps "grading" what it did and supplying corrections — this fixes the classic imitation-learning problem where a policy drifts into states the expert never demonstrated.
3. **Phase 4 (pure MLP):** MPPI is switched off. The trained network runs the robot on its own.

This is the standard **"distill an expensive expert controller into a cheap neural network"** recipe. MPPI is slow (150 parallel physics rollouts every step!) but doesn't need training. A trained MLP is instant at runtime but needs data — DAgger is what makes that data good.

---

## 1. Imports & model loading

```julia
using MuJoCo, LinearAlgebra, Random, Statistics, Base.Threads
mj_model = load_model("go2/scene.xml")
mj_data  = init_data(mj_model)
include("mlp_policy.jl")
```
- Loads the Unitree Go2 quadruped MuJoCo model/scene.
- `mj_data` is the *mutable simulation state* (positions, velocities, etc.) — MuJoCo separates the static model (`mj_model`) from the per-simulation state (`mj_data`).
- `mlp_policy.jl` is a separate file (not shown) that presumably defines `bc_policy`, `bc_history_len`, `bc_predict`, `bc_prepare_dataset`, `train_bc_policy!` — the neural-network side of things ("bc" = behavior cloning).

## 2. Core constants

```julia
const nu = mj_model.nu   # number of actuators (12 for a quadruped: 3 joints x 4 legs)
const H  = 40             # MPPI planning horizon (steps looked ahead)
const K  = 150             # number of sampled trajectories per MPPI update
const λ  = 0.1             # (defined but note: mppi_update! recomputes its own λ_eff — this const isn't actually used there)
const KP = 80.0            # PD position-gain
const KD = 4.0             # PD velocity-gain (damping)
```
**Intuition:** MPPI doesn't optimize control torques directly. It optimizes **target joint positions** for a PD controller, which then converts those targets into torques. This is much smoother/safer than raw torque search.

`H` and `K` are the classic MPPI knobs: `H` = how far into the future you simulate each candidate plan, `K` = how many random plans you try. Bigger `K` = better search, more compute.

```julia
const HOME_POS = Float64[0.0, 0.9, -1.8, ...]  # repeated x4 legs
```
A hand-tuned "default standing pose" for each of the 12 joints (hip, thigh, calf angles) — used as (a) the reset pose, (b) the fallback plan tail, and (c) a regularizer target in the cost function.

```julia
const noise_sigma = Float64[0.08, 0.15, 0.15, ...]
```
Per-joint exploration noise standard deviation used to perturb candidate trajectories in MPPI. Hip joints (0.08) get less noise than thigh/calf (0.15) — probably because hip motion destabilizes balance more easily.

```julia
const TORQUE_LIMIT = Float64[23.7, 23.7, 45.43, ...]
```
Actuator torque limits (Nm) — the calf joint (3rd in each triplet) has a much bigger motor (45.43 Nm) than hip/thigh (23.7 Nm), matching real Go2 hardware specs.

```julia
const JOINT_LOWER / JOINT_UPPER
```
Physical joint angle limits (radians), used both to clamp commands and to penalize approaching the limits in the cost function.

```julia
const U_global = zeros(nu, H)
dataset = []
```
`U_global` is the **current best control plan** — a `12 × 40` matrix of planned joint targets for the next 40 steps. This persists across control loop calls ("warm-starting" MPPI — you don't replan from scratch each time, you shift the previous plan forward).

`dataset` accumulates `(state_history, target_action)` training pairs for the MLP.

---

## 3. The cost function — this *is* the robot's objective

```julia
function cost(qpos, qvel, pos_target)
```
This is evaluated at every simulated timestep of every sampled MPPI rollout. It's a weighted sum of penalties — this is where "what do we want the robot to do" gets encoded numerically. Let's go term by term:

- **Catastrophic fall guard:**
  ```julia
  if qpos[3] < 0.20
      return 1_000_000.0
  end
  ```
  `qpos[3]` is the base height (z-coordinate of the free joint). Below 20cm = essentially fallen → astronomically bad cost, kills that rollout's viability instantly.

- **Height tracking:**
  ```julia
  height_cost = h <= target_height ? 30000.0 * h_diff^2 : 60000.0 * h_diff^2
  ```
  Quadratic penalty toward a 0.38m standing height, but **asymmetric**: being too high is penalized 2x harder than being too low. That's a deliberate bias — probably because overshooting height risks the robot toppling/overextending legs, so the cost function is more conservative about being too tall than too short.

- **Vertical velocity damping:** `5000.0 * qvel[3]^2` — discourages bouncing.

- **Orientation (staying upright):**
  ```julia
  qw, qx, qy, qz = qpos[4:7]
  gravity_z = 1.0 - 2.0*(qx^2 + qy^2)
  quat_cost = 150000.0 * (1.0 - clamp(gravity_z, -1, 1))^2
  ```
  This is a standard trick: for a unit quaternion `(w,x,y,z)`, the z-component of the "up" vector after rotation is `1 - 2(x²+y²)`. When the robot is perfectly upright, this equals 1. As it tips over, it shrinks toward 0 or negative. Squaring `(1 - gravity_z)` heavily punishes tipping — and with a huge weight (150000), this is one of the dominant cost terms, i.e. "staying upright" matters more than almost anything else.

  `yaw_cost = 50000.0 * qz^2` separately penalizes yaw rotation (spinning around vertical axis) using the quaternion z-component directly — discourages the robot twisting instead of walking straight.

- **Joint limit avoidance:**
  ```julia
  margin = 0.15
  if qi < lo  joint_range_cost += 5000*(lo-qi)^2 ...
  ```
  Soft barrier: starts penalizing once a joint gets within 0.15 rad of its hard limit, growing quadratically. Keeps MPPI from planning trajectories that would slam into mechanical stops.

- **Leg symmetry:** `sym_cost` penalizes left/right joint-angle differences (front-left vs front-right, rear-left vs rear-right). This discourages "lopsided" or "one leg doing all the work" gaits.

- **Pose regularization:** `pose_reg_cost = 2000 * sum((qpos[8:19] .- HOME_POS).^2)` — a gentle pull back toward the nominal standing pose, preventing MPPI from finding weird, unnatural joint configurations that technically satisfy other costs.

- **Forward velocity tracking:**
  ```julia
  vel_cost = vel_x <= 0.3 ? 50000*(vel_x-0.3)^2 : 120000*(vel_x-0.3)^2
  ```
  Target forward speed 0.3 m/s. Again asymmetric — overspeeding is punished more than 2x as hard as underspeeding, presumably because going too fast risks a forward faceplant.

- **Lateral velocity & joint velocity:** small penalties discouraging sideways drift and jerky/fast joint motion (energy/smoothness regularizers).

**Intuition to take away:** designing an MPPI cost is basically reward-shaping — every behavior you want (height, uprightness, forward speed, symmetry, smoothness) needs its own explicit term, and *relative weights* determine priorities. Here, "don't fall over" (quat_cost, the 1,000,000 hard cutoff) dominates everything else, as it should.

---

## 4. `rollout` — the sampling engine of MPPI

```julia
function rollout(m, d, U, noise)
    @threads for k in 1:K
        d_copy = init_data(m)
        d_copy.qpos .= d.qpos; d_copy.qvel .= d.qvel
        ...
        for t in 1:H
            pos_target = U[:,t] + noise[:,t,k]
            ...
            mj_step(m, d_copy)
            cost_sum += cost(...)
        end
        costs[k] = cost_sum
    end
end
```
**This is the heart of MPPI.** For each of the `K=150` samples (run in parallel via `@threads`):
1. Clone the current real robot state into a private scratch `d_copy` (so the 150 simulations don't interfere with each other or the real state).
2. Add random noise to the current nominal plan `U` at every timestep → get a candidate 40-step trajectory of joint targets.
3. Convert those targets into torques via **PD control**:
   ```julia
   raw_torques = KP .* (pos_target .- joint_pos) .- KD .* joint_vel
   ```
   Classic PD: proportional term pulls toward target position, derivative term (using *current* velocity, not target — this is a "PD on measurement" variant) damps oscillation.
4. **Joint reindexing** (`ctrl_torques[1:3] .= raw_torques[4:6]` etc.) — this remaps leg order because MuJoCo's actuator order (FR, FL, RR, RL or similar) differs from the FL/FR/RL/RR order used for `HOME_POS`/noise arrays in this code. This is a hardware/model-specific quirk you'd need to double check against your actual `scene.xml` actuator ordering.
5. Clamp to torque limits, step physics forward one timestep (`mj_step`), accumulate cost.
6. After 40 steps, `costs[k]` = total cost of that entire candidate trajectory.

You end up with 150 candidate future trajectories, each with a total cost — this is literally "shooting" 150 possible futures and seeing which ones look good.

---

## 5. `mppi_update!` — turning rollouts into a control decision

```julia
function mppi_update!(m, d)
    noise = randn(...) .* noise_sigma   # sample K*H random perturbations
    costs, trajectories = rollout(m, d, U_global, noise)
```
Sample noise, roll out all K candidates.

```julia
    β = minimum(costs)
    cost_std = std(costs) + 1e-6
    λ_eff = 0.15 * cost_std
    weights = exp.(-1/λ_eff * (costs .- β))
    weights ./= sum(weights) + 1e-10
```
**This is the "path integral" weighting step**, the mathematical core of MPPI:
- Subtract the best cost `β` (numerical stability — prevents `exp` overflow).
- Turn costs into weights via a softmin: low-cost trajectories get exponentially higher weight.
- `λ_eff` is the **temperature**, and notably it's *adaptive* here (`0.15 * cost_std`) rather than the fixed `λ=0.1` constant defined earlier — this rescales sensitivity to the actual spread of costs seen this iteration, so the softmin doesn't become too sharp or too flat depending on how much the rollouts vary. This is a nice practical trick beyond textbook MPPI.
- Normalize into a probability distribution over the 150 samples.

```julia
    for t in 1:H
        weighted_noise = sum(weights[k] * noise[:,t,k] for k in 1:K)
        U_global[:,t] .= clamp.(U_global[:,t] + weighted_noise, JOINT_LOWER, JOINT_UPPER)
    end
```
The new plan = old plan + a **weighted average of all the noise vectors**, weighted by how good each noisy trajectory turned out. This is the "path integral" — instead of gradient descent, you're doing a soft, weighted majority vote across random samples. Good samples pull the plan toward themselves; bad samples are effectively ignored.

```julia
    raw_torques = KP .* (U_global[:,1] .- joint_pos) .- KD .* joint_vel
    ... d.ctrl .= clamp.(ctrl_torques, ...)
```
Apply only the **first timestep** of the newly updated plan to the real robot (standard receding-horizon / MPC pattern: plan `H` steps ahead, execute 1 step, replan).

```julia
    U_global[:,1:end-1] .= U_global[:,2:end]
    U_global[:,end] .= HOME_POS
```
**Warm-starting / shifting:** slide the plan one step forward (what was planned for t=2 becomes the new t=1 guess) and pad the freed-up last slot with `HOME_POS`. This means each new MPPI call doesn't start from scratch — it refines a plan that's already roughly right, which is much more sample-efficient than resampling from zero every time.

---

## 6. State history buffer

```julia
const state_history = Vector{Float32}[]
function get_history_vector(current_state)
    push!(state_history, current_state)
    while length > bc_history_len; popfirst!(...) end
    while length < bc_history_len; pushfirst!(...) end
    return vcat(state_history...)
end
```
The MLP policy isn't given just the current instantaneous state — it's given a **stacked window of the last `bc_history_len` states**, concatenated into one long vector. This is common for policies trained on legged robots because a single snapshot of joint angles/velocities doesn't reveal *dynamics* (e.g., is the robot currently accelerating, oscillating?) — history gives implicit velocity/phase information. The padding logic ensures the buffer is always full length, even right after a reset (by repeating the first state).

`clear_history_buffer!` resets this queue by filling it entirely with the post-reset state — important so a fall/reset doesn't leave stale pre-fall history polluting the next episode.

---

## 7. `collect_dataset!` — building supervised-learning training pairs

```julia
function collect_dataset!(dataset, history_vec, U_opt)
    target_residual = U_opt .- HOME_POS
    push!(dataset, (history_vec, vec(Float32.(target_residual))))
end
```
Instead of training the MLP to predict raw joint-target trajectories, it predicts the **residual from HOME_POS**. This is a very standard imitation-learning trick: predicting "deviation from a known-good default" is an easier regression problem than predicting absolute values from scratch, and it biases an undertrained network toward the safe home pose rather than toward zero/garbage.

Each dataset entry = `(stacked state history) → (residual target trajectory that MPPI decided was good)`.

---

## 8. `reset_robot!`

```julia
MuJoCo.mj_resetData(m, d)
d.qpos[1:3] .= [0,0,0.27]   # base position, dropped from slightly above ground
d.qpos[4:7] .= [1,0,0,0]     # identity quaternion = upright
d.qpos[8:19] .= HOME_POS     # joints at home pose
d.qvel .= 0
for _ in 1:10
    ... mj_step(m,d)   # settle onto the ground with PD holding home pose
end
for t in 1:H; U_global[:,t] .= HOME_POS; end
clear_history_buffer!(...)
```
Standard episode-reset: re-zero the sim, place robot at 0.27m height in home pose, run 10 physics substeps of pure PD-hold so it settles onto the ground under gravity/contact before control resumes, reset the MPPI plan buffer to all-home, and clear the history queue so the network doesn't see cross-episode contamination.

---

## 9. The controller state machine — this is the DAgger orchestration

```julia
const DATA_ITERS   = 500
const DAGGER_ROUNDS   = 2
const DAGGER_STEPS    = 200
const FINAL_MLP_ITERS = 500
```
Total run = 500 (pure MPPI) + 2×200 (DAgger) + 500 (pure MLP) = 1400 controller calls.

```julia
function compute_phase_boundary()
    return DATA_ITERS + DAGGER_ROUNDS * DAGGER_STEPS
end
```
Computes the iteration index where DAgger ends and pure-MLP execution begins (500 + 400 = 900).

`visual_controller!(m, d)` is called once per simulation step by the visualizer (`controller=visual_controller!` in `visualise!` at the bottom). It's a big `if/elseif`-style dispatcher based on `vis_iter`:

### Phase 1–2 (`iter <= DATA_ITERS`, the `else` branch at the bottom of the function)
```julia
costs, trajs = mppi_update!(m, d)
collect_dataset!(dataset, hist_vec, U_global)
```
Pure MPPI drives the robot: every step, run the full 150-sample MPPI update, apply it, and log `(history, MPPI's plan)` as a training example. Falls trigger `reset_robot!`.

At `iter == DATA_ITERS` (step 500):
```julia
retrain_policy!()
policy_ready = true
```
The very first training of the MLP happens here, on pure-expert (MPPI-only) data — this is standard **behavior cloning (BC)** initialization before DAgger kicks in.

### Phase 3 — DAgger rounds (`DATA_ITERS < iter <= dagger_end`)
```julia
if policy_ready && mod(step_in_round-1, EXEC_STEPS)==0
    traj = bc_predict(bc_policy, hist_vec, ...)
    U_global .= clamp.(traj, ...)
end
```
Now the **MLP is in the driver's seat** — its prediction overwrites `U_global` every `EXEC_STEPS=4` steps (receding horizon at the policy level too). The robot moves under **the network's own control**, so it visits whatever states its own (possibly imperfect) policy leads it into — including states the original MPPI-only dataset never saw (e.g., near-falls, drift).

Then, crucially:
```julia
saved_U = copy(U_global)
for t in 1:H; U_global[:,t] .= 0.7*saved_U[:,t] .+ 0.3*HOME_POS; end
for _ in 1:3; mppi_update!(m,d); end
expert_U = copy(U_global)
U_global .= saved_U   # restore — don't let this expert query affect the real robot's action!
collect_dataset!(dataset, hist_vec, expert_U)
```
This is the **DAgger correction step**: even though the network is actually controlling the robot, MPPI is asked "given the state the network just put us in, what *should* the plan be?" (initialized from a blend of the network's own plan and home, then refined via 3 MPPI iterations). That expert answer is logged as a new training pair — labeling states the *policy* visits with the *expert's* corrective action. Note `U_global .= saved_U` restores the network's actual plan afterward so this "asking the expert for advice" doesn't secretly hijack the robot's real trajectory — a good implementation detail, since otherwise this would just silently degrade back into pure MPPI.

This is exactly the DAgger algorithm: **collect labels from the expert, but on the state distribution induced by the learner**, which is what fixes covariate-shift/compounding-error problems that plain behavior cloning suffers from.

At the end of each round (`step_in_round == DAGGER_STEPS`):
```julia
retrain_policy!()
if current_round == DAGGER_ROUNDS
    policy_ready = true
end
```
Retrain the MLP on the *aggregated* dataset (original MPPI data + all DAgger-corrected data so far) — this is the "aggregate" in "Dataset Aggregation."

### Phase 4 — pure MLP (`iter > dagger_end`)
```julia
traj = bc_predict(bc_policy, hist_vec, ...)
U_global .= clamp.(traj, ...)
```
No more MPPI at all. The trained network fully controls the robot for 500 steps, logging height/velocity every 10 steps so you can visually judge whether training succeeded. This is the "final exam."

---

## 10. `retrain_policy!` — the actual learning step

```julia
filtered = filter(d -> d[1][3] >= 0.28, dataset)
```
Interesting data-cleaning step: it drops training samples where the height channel in the *state history* (`d[1][3]`, presumably the height field of the most recent stacked state) is below 0.28m. In other words, **it throws out data from near-fall states**, so the network isn't taught to imitate MPPI's fall-recovery scrambling — only "healthy standing/walking" data is kept.

```julia
Q = hcat([d[1] for d in filtered]...)
U_ds = hcat([d[2] for d in filtered]...)
p4_μ_q, p4_σ_q = mean/std(Q)
p4_μ_u, p4_σ_u = mean/std(U_ds)
X_state, Y_traj = bc_prepare_dataset(filtered, μ_q, σ_q, μ_u, σ_u)
train_bc_policy!(bc_policy, X_state, Y_traj; epochs=120, batchsize=32, lr=5e-4)
```
Standard supervised learning recipe: compute normalization statistics (mean/std) for both inputs (state histories) and outputs (residual action targets) — this is essential for MLP training stability — then train for 120 epochs with batch size 32 and learning rate 5e-4. Note the normalization stats (`p4_μ_q` etc.) are recomputed and overwritten *every* retrain call, so later DAgger-round normalization reflects the growing aggregated dataset.

---

## 11. Wiring it up

```julia
init_visualiser()
visualise!(mj_model, mj_data; controller=visual_controller!)
```
Starts the MuJoCo viewer and hands it `visual_controller!` as the per-timestep callback — everything above runs driven by this single callback being invoked once per simulated frame.

---

## Key concepts to internalize

| Concept | Where in code | Intuition |
|---|---|---|
| **MPPI** | `rollout` + `mppi_update!` | Sample many random control sequences, weight them by how well they did (softmin), take a weighted average as the new plan. No gradients needed — works even through hard contact dynamics that are non-differentiable. |
| **Receding horizon (MPC)** | Plan `H=40` steps, apply only step 1, shift and replan | You always act on your *most recent* best guess, re-planned every single timestep, so errors get corrected constantly. |
| **PD control as the actuation layer** | `KP .* (target - pos) .- KD .* vel` | MPPI/MLP output *joint position targets*, not torques — a low-level PD loop converts targets to safe, smooth torques. Decouples "what pose to reach for" (planning) from "how to physically get there" (low-level control). |
| **Behavior cloning (BC)** | `retrain_policy!` after Phase 1 | Supervised learning: train a network to imitate expert (state → action) pairs. |
| **DAgger** | Phase 3 loop | Fixes BC's blind spot: BC only ever sees expert-visited states, so a BC-only policy compounds small errors into big ones once it drifts off-distribution at test time. DAgger re-queries the expert on states the *learner itself* visits, closing that gap iteratively. |
| **Residual/delta learning** | `target_residual = U_opt - HOME_POS` | Predicting a small correction from a known-good baseline is numerically easier and safer than predicting raw absolute targets. |
| **History-conditioned policy** | `state_history` buffer | Gives the MLP short-term temporal context (implicit velocity/phase info) instead of a memoryless single-frame input. |

If you want, I can also help you trace through *why* specific weight magnitudes were chosen in the cost function, or sketch the expected shape of `mlp_policy.jl` (which isn't included here) so you can see how `bc_predict`/`train_bc_policy!` likely work.