using MuJoCo
using LinearAlgebra
using Random
using Statistics
using Base.Threads

mj_model = load_model("go2/scene.xml")
mj_data  = init_data(mj_model)
include("diffusion1_model.jl")

const K  = 50
const λ  = 0.1
const KP = 80.0    # PD proportional gain  (80×0.3rad ≈ hip limit → linear in normal range)
const KD = 4.0     # PD derivative gain    (damping ratio preserved)

# Go2 stable standing pose: [hip, thigh, knee] × 4 legs
# MuJoCo qpos joint order in XML: FL, FR, RL, RR
const HOME_POS = Float64[
    0.0, 0.9, -1.8,  # FL
    0.0, 0.9, -1.8,  # FR
    0.0, 0.9, -1.8,  # RL
    0.0, 0.9, -1.8   # RR
]

const noise_sigma = Float64[
    0.08, 0.15, 0.15,
    0.08, 0.15, 0.15,
    0.08, 0.15, 0.15,
    0.08, 0.15, 0.15
]

# Per-joint torque limits from MJCF: hip/thigh = 23.7, knee = 45.43
const TORQUE_LIMIT = Float64[
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43
]

# Per-joint position limits from MJCF  [hip, thigh, knee] × 4 legs
# MuJoCo qpos joint order: 1..3=FL, 4..6=FR, 7..9=RL, 10..12=RR
# FL/FR thigh = front_hip (-1.5708, 3.4907)
# RL/RR thigh = back_hip  (-0.5236, 4.5379)
const JOINT_LOWER = Float64[
    -1.0472, -1.5708, -2.7227,  # FL
    -1.0472, -1.5708, -2.7227,  # FR
    -1.0472, -0.5236, -2.7227,  # RL
    -1.0472, -0.5236, -2.7227   # RR
]
const JOINT_UPPER = Float64[
     1.0472,  3.4907, -0.83776, # FL
     1.0472,  3.4907, -0.83776, # FR
     1.0472,  4.5379, -0.83776, # RL
     1.0472,  4.5379, -0.83776  # RR
]

# U_global now holds JOINT POSITION TARGETS not torques
const U_global = zeros(nu, H)
dataset = []

# ── Cost ─────────────────────────────────────────────────────────────
function cost(qpos, qvel, pos_target)
    target_height = 0.38   # Natural dynamic walking height
    target_vel_x  = 0.3
    target_quat   = [1.0, 0.0, 0.0, 0.0]

    if qpos[3] < 0.20
        return 1_000_000.0
    end

    # Height penalty centered at 0.38m
    h = qpos[3]
    h_diff = h - target_height
    height_cost = if h <= target_height
        30000.0 * h_diff^2
    else
        60000.0 * h_diff^2
    end

    # Vertical body velocity penalty
    z_vel_cost = 5000.0 * qvel[3]^2

    # Orientation via Projected Gravity Vector (Roll & Pitch)
    qw, qx, qy, qz = qpos[4], qpos[5], qpos[6], qpos[7]
    gravity_z = 1.0 - 2.0 * (qx^2 + qy^2)
    quat_cost = 150000.0 * (1.0 - clamp(gravity_z, -1.0, 1.0))^2

    # Yaw Heading Lock — prevents turning left/right (qz = qpos[7] controls yaw)
    yaw_cost = 50000.0 * qz^2

    # Soft joint-limit penalty
    joint_range_cost = 0.0
    margin = 0.15
    for i in 1:12
        qi = qpos[7+i]
        lo = JOINT_LOWER[i] + margin
        hi = JOINT_UPPER[i] - margin
        if qi < lo
            joint_range_cost += 5000.0 * (lo - qi)^2
        elseif qi > hi
            joint_range_cost += 5000.0 * (qi - hi)^2
        end
    end

    # Leg symmetry cost: FL vs FR and RL vs RR
    sym_cost = 300.0 * (
        sum((qpos[8:10]  .- qpos[11:13]).^2) +
        sum((qpos[14:16] .- qpos[17:19]).^2)
    )

    # Forward velocity — unified quadratic penalty (target_vel_x = 0.3 m/s)
    # Heavy penalty whenever speed is less than 0.3 m/s or negative
    vel_x = qvel[1]
    vel_cost = if vel_x <= target_vel_x
        50000.0 * (vel_x - target_vel_x)^2        # Strong forward pull towards 0.3 m/s
    else
        100000.0 * (vel_x - target_vel_x)^2       # Heavy penalty for overspeeding >0.3 m/s to prevent forward pitch collapse
    end

    # Lateral drift penalty
    lateral_cost = 1500.0 * qvel[2]^2

    # Joint velocity smoothness
    jvel_cost = 5.0 * sum(qvel[7:18].^2)

    return height_cost + z_vel_cost + quat_cost + yaw_cost + joint_range_cost + sym_cost + vel_cost + lateral_cost + jvel_cost
end

# ── Rollout ───────────────────────────────────────────────────────────
function rollout(m::Model, d::Data, U::Matrix{Float64}, noise::Array{Float64,3})
    costs = zeros(K)
    trajectories = Array{Float64,3}(undef, K, nu, H)

    @threads for k in 1:K
        d_copy = init_data(m)
        d_copy.qpos .= d.qpos
        d_copy.qvel .= d.qvel
        cost_sum = 0.0

        for t in 1:H
            # MPPI samples joint position targets
            pos_target = vec(U[:, t] + noise[:, t, k])

            # Convert joint positions (FL, FR, RL, RR) to actuator torques (FR, FL, RR, RL)
            # Motor index map: FR=[1,2,3], FL=[4,5,6], RR=[7,8,9], RL=[10,11,12]
            # Joint index map: FL=[1,2,3], FR=[4,5,6], RL=[7,8,9], RR=[10,11,12]
            joint_pos = d_copy.qpos[8:19]
            joint_vel = d_copy.qvel[7:18]
            raw_torques = KP .* (pos_target .- joint_pos) .- KD .* joint_vel
            
            # Map joints (FL, FR, RL, RR) -> actuators (FR, FL, RR, RL)
            ctrl_torques = zeros(12)
            ctrl_torques[1:3]   .= raw_torques[4:6]   # FR
            ctrl_torques[4:6]   .= raw_torques[1:3]   # FL
            ctrl_torques[7:9]   .= raw_torques[10:12] # RR
            ctrl_torques[10:12] .= raw_torques[7:9]   # RL

            d_copy.ctrl .= clamp.(ctrl_torques, -TORQUE_LIMIT, TORQUE_LIMIT)
            mj_step(m, d_copy)

            cost_sum += cost(d_copy.qpos, d_copy.qvel, pos_target)
            trajectories[k, :, t] .= pos_target
        end
        costs[k] = cost_sum
    end

    return costs, trajectories
end

# ── MPPI Update ───────────────────────────────────────────────────────
function mppi_update!(m::Model, d::Data)
    noise = zeros(nu, H, K)
    for k in 1:K
        for t in 1:H
            noise[:, t, k] .= randn(nu) .* noise_sigma
        end
    end

    costs, trajectories = rollout(m, d, U_global, noise)

    β = minimum(costs)
    cost_std = std(costs) + 1e-6
    λ_eff = 0.15 * cost_std       # Sharper temperature — selects top performing trajectories
    weights = exp.(-1 / λ_eff * (costs .- β))
    weights ./= sum(weights) + 1e-10

    for t in 1:H
        weighted_noise = sum(weights[k] * noise[:, t, k] for k in 1:K)
        # Clamp U_global to valid mechanical joint limits, not generic (-π, π)
        U_global[:, t] .= clamp.(U_global[:, t] + weighted_noise, JOINT_LOWER, JOINT_UPPER)
    end

    # Apply via PD to real robot (mapping joints FL,FR,RL,RR -> actuators FR,FL,RR,RL)
    joint_pos = d.qpos[8:19]
    joint_vel = d.qvel[7:18]
    raw_torques = KP .* (U_global[:, 1] .- joint_pos) .- KD .* joint_vel
    
    ctrl_torques = zeros(12)
    ctrl_torques[1:3]   .= raw_torques[4:6]   # FR
    ctrl_torques[4:6]   .= raw_torques[1:3]   # FL
    ctrl_torques[7:9]   .= raw_torques[10:12] # RR
    ctrl_torques[10:12] .= raw_torques[7:9]   # RL

    d.ctrl .= clamp.(ctrl_torques, -TORQUE_LIMIT, TORQUE_LIMIT)

    U_global[:, 1:end-1] .= U_global[:, 2:end]
    U_global[:, end] .= HOME_POS

    return costs, trajectories
end

# ── Non-Invasive MPPI Expert Querying ──────────────────────────────────
function query_mppi_expert(m::Model, state_qpos::Vector{Float64}, state_qvel::Vector{Float64}, warm_U::Matrix{Float64})
    d_query = init_data(m)
    d_query.qpos .= state_qpos
    d_query.qvel .= state_qvel

    # Warm start directly from current trajectory warm_U (no static HOME_POS decay)
    U_expert = copy(warm_U)

    # Run 15 refinement iterations on isolated d_query copy so MPPI fully converges
    for _ in 1:15
        noise = zeros(nu, H, K)
        for k in 1:K, t in 1:H
            noise[:, t, k] .= randn(nu) .* noise_sigma
        end
        costs, trajectories = rollout(m, d_query, U_expert, noise)
        β = minimum(costs)
        cost_std = std(costs) + 1e-6
        λ_eff = 0.15 * cost_std
        weights = exp.(-1 / λ_eff * (costs .- β))
        weights ./= sum(weights) + 1e-10

        for t in 1:H
            weighted_noise = sum(weights[k] * noise[:, t, k] for k in 1:K)
            U_expert[:, t] .= clamp.(U_expert[:, t] + weighted_noise, JOINT_LOWER, JOINT_UPPER)
        end
    end

    return U_expert
end

# ── State History Queue Buffer (History-Conditioned Diffusion) ───────────
const state_history = Vector{Float32}[]

function get_history_vector(current_state::AbstractVector{Float32})
    curr_vec = vec(current_state)
    push!(state_history, curr_vec)
    while length(state_history) > diff_history_len
        popfirst!(state_history)
    end
    while length(state_history) < diff_history_len
        pushfirst!(state_history, state_history[1])
    end
    return vcat(state_history...)
end

function clear_history_buffer!(initial_state::AbstractVector{Float32})
    empty!(state_history)
    init_vec = vec(initial_state)
    for _ in 1:diff_history_len
        push!(state_history, init_vec)
    end
end

# ── Dataset (Saves true converged U_global trajectory) ─────────────────
function collect_dataset!(dataset, history_vec::Vector{Float32}, U_opt::Matrix{Float64})
    push!(dataset, (
        history_vec,
        vec(Float32.(U_opt))  # True MPPI weighted optimal trajectory (nu x H)
    ))
end

# ── Reset (Natural XML Keyframe Stance) ─────────────────────────────────
function reset_robot!(m, d)
    MuJoCo.mj_resetData(m, d)
    # Set natural standing keyframe from XML: height 0.27m on ground
    d.qpos[1:3]  .= [0.0, 0.0, 0.27]
    d.qpos[4:7]  .= [1.0, 0.0, 0.0, 0.0]
    d.qpos[8:19] .= HOME_POS
    d.qvel       .= 0.0
    
    # Pre-settle robot on ground with PD control
    for _ in 1:10
        raw_torques = KP .* (HOME_POS .- d.qpos[8:19]) .- KD .* d.qvel[7:18]
        ctrl_torques = zeros(12)
        ctrl_torques[1:3]   .= raw_torques[4:6]   # FR
        ctrl_torques[4:6]   .= raw_torques[1:3]   # FL
        ctrl_torques[7:9]   .= raw_torques[10:12] # RR
        ctrl_torques[10:12] .= raw_torques[7:9]   # RL
        d.ctrl .= clamp.(ctrl_torques, -TORQUE_LIMIT, TORQUE_LIMIT)
        mj_step(m, d)
    end
    clear_history_buffer!(vec(Float32.(vcat(d.qpos, d.qvel))))
end

# ── Phase 1: Data Collection (visualised) ─────────────────────────────
reset_robot!(mj_model, mj_data)
for t in 1:H; U_global[:, t] .= HOME_POS; end
global fall_count = 0
global vis_iter = 0

# Metric logging
log_iter       = Int[]
log_min_cost   = Float64[]
log_height     = Float64[]
log_vel_x      = Float64[]
log_height_c   = Float64[]
log_quat_c     = Float64[]
log_joint_c    = Float64[]   # joint_range_cost
log_sym_c      = Float64[]   # symmetry cost
log_vel_c      = Float64[]
log_lateral_c  = Float64[]
log_jvel_c     = Float64[]
log_n_sat      = Int[]
log_max_torque = Float64[]

const DATA_ITERS   = 500     # Collect 500 MPPI expert iterations for well-converged dataset
const EXEC_STEPS   = 4       # Receding horizon: re-predict every 4 steps

# DAgger configuration
const DAGGER_ROUNDS        = 2     # Number of DAgger correction rounds
const DAGGER_STEPS         = 200   # Steps per DAgger round (Diffusion policy acts, MPPI labels)
const FINAL_DIFFUSION_ITERS = 1000  # Final pure Diffusion execution after all DAgger rounds (1000 steps)

# Globals for Phase 4 normalization stats & training state
global policy_ready = false
global dagger_round = 0
global dagger_step_in_round = 0
global phase4_μ_q = nothing
global phase4_σ_q = nothing
global phase4_μ_u = nothing
global phase4_σ_u = nothing
global phase4_α   = nothing

function compute_phase_boundary()
    dagger_total = DAGGER_ROUNDS * DAGGER_STEPS
    return DATA_ITERS + dagger_total
end

function retrain_diffusion_policy!()
    global phase4_μ_q, phase4_σ_q, phase4_μ_u, phase4_σ_u, phase4_α

    # Filter out fallen states (most recent frame height in history < 0.28m)
    filtered_dataset = filter(d -> d[1][end-mj_model.nv-mj_model.nq+3] >= 0.28, dataset)
    println("✓ Retraining Diffusion model on aggregated dataset: $(length(dataset)) total steps. Clean non-fallen samples: $(length(filtered_dataset))")

    Q = hcat([d[1] for d in filtered_dataset]...)
    U_ds = hcat([d[2] for d in filtered_dataset]...)
    phase4_μ_q = Float32.(mean(Q, dims=2))
    phase4_σ_q = Float32.(std(Q,  dims=2) .+ 1e-6)
    phase4_μ_u = Float32.(mean(U_ds, dims=2))
    phase4_σ_u = Float32.(std(U_ds,  dims=2) .+ 1e-6)
    X_state, X_ctrl = prepare_dataset(filtered_dataset, phase4_μ_q, phase4_σ_q, phase4_μ_u, phase4_σ_u)

    # Train Diffusion Model on aggregated MPPI Trajectories
    train_diffusion!(model, X_state, X_ctrl; epochs=100, batchsize=64, lr=1e-3)

    # Precompute noise schedule
    β_sched = Float32.(collect(LinRange(1e-4, 0.18, T_diff)))
    phase4_α = cumprod(1.0f0 .- β_sched)
end

function visual_controller!(m, d)
    global vis_iter, fall_count, policy_ready
    global dagger_round, dagger_step_in_round
    global phase4_μ_q, phase4_σ_q, phase4_μ_u, phase4_σ_u, phase4_α

    vis_iter += 1
    iter = vis_iter

    dagger_end = compute_phase_boundary()
    total_end  = dagger_end + FINAL_DIFFUSION_ITERS

    # Update state history buffer with current frame
    curr_state_vec = vec(Float32.(vcat(d.qpos, d.qvel)))
    hist_vec = get_history_vector(curr_state_vec)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: Pure Diffusion Policy (Receding Horizon + DAgger Trained)
    # ═══════════════════════════════════════════════════════════════════
    if iter > dagger_end
        if iter > total_end
            return
        end

        final_step = iter - dagger_end

        # Re-predict trajectory from current state history every EXEC_STEPS
        if policy_ready && mod(final_step - 1, EXEC_STEPS) == 0
            norm_state = (hist_vec .- phase4_μ_q) ./ phase4_σ_q
            u_ws = ddim_sample(model, norm_state, phase4_α, T_diff; num_steps=20)
            μ_u_mat = reshape(phase4_μ_u, nu, H)
            σ_u_mat = reshape(phase4_σ_u, nu, H)
            global U_global .= clamp.((u_ws .* σ_u_mat) .+ μ_u_mat, JOINT_LOWER, JOINT_UPPER)
        end

        # Apply PD control from U_global[:, 1]
        joint_pos = d.qpos[8:19]
        joint_vel = d.qvel[7:18]
        raw_torques = KP .* (U_global[:, 1] .- joint_pos) .- KD .* joint_vel

        ctrl_torques = zeros(12)
        ctrl_torques[1:3]   .= raw_torques[4:6]   # FR
        ctrl_torques[4:6]   .= raw_torques[1:3]   # FL
        ctrl_torques[7:9]   .= raw_torques[10:12] # RR
        ctrl_torques[10:12] .= raw_torques[7:9]   # RL
        d.ctrl .= clamp.(ctrl_torques, -TORQUE_LIMIT, TORQUE_LIMIT)

        # Shift trajectory horizon forward
        U_global[:, 1:end-1] .= U_global[:, 2:end]
        U_global[:, end] .= HOME_POS

        if mod(final_step, 10) == 0
            println("Phase4 [Diffusion Only] Step $final_step | height = $(round(d.qpos[3], digits=3)) | vel_x = $(round(d.qvel[1], digits=3))")
        end

        # Fall detection during Phase 4
        qw, qx, qy, qz = d.qpos[4], d.qpos[5], d.qpos[6], d.qpos[7]
        gravity_z = 1.0 - 2.0 * (qx^2 + qy^2)
        if d.qpos[3] < 0.22 || gravity_z < 0.64
            fall_count += 1
            println("--- Phase4 Fall #$fall_count — resetting ---")
            reset_robot!(m, d)
        end

        return
    end

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: DAgger Correction Rounds for Diffusion Policy
    # ═══════════════════════════════════════════════════════════════════
    if iter > DATA_ITERS && iter <= dagger_end
        dagger_offset = iter - DATA_ITERS
        current_round = div(dagger_offset - 1, DAGGER_STEPS) + 1
        step_in_round = mod(dagger_offset - 1, DAGGER_STEPS) + 1

        if step_in_round == 1 && current_round != dagger_round
            dagger_round = current_round
            println("\n=== DAgger Round $dagger_round / $DAGGER_ROUNDS: Diffusion policy acts, MPPI relabels ($DAGGER_STEPS steps) ===")
            reset_robot!(m, d)
            for t in 1:H; U_global[:, t] .= HOME_POS; end
        end

        # Diffusion policy rollouts state history action prediction
        if policy_ready && mod(step_in_round - 1, EXEC_STEPS) == 0
            norm_state = (hist_vec .- phase4_μ_q) ./ phase4_σ_q
            u_ws = ddim_sample(model, norm_state, phase4_α, T_diff; num_steps=20)
            μ_u_mat = reshape(phase4_μ_u, nu, H)
            σ_u_mat = reshape(phase4_σ_u, nu, H)
            global U_global .= clamp.((u_ws .* σ_u_mat) .+ μ_u_mat, JOINT_LOWER, JOINT_UPPER)
        end

        # Execute actions on robot
        joint_pos = d.qpos[8:19]
        joint_vel = d.qvel[7:18]
        raw_torques = KP .* (U_global[:, 1] .- joint_pos) .- KD .* joint_vel
        ctrl_torques = zeros(12)
        ctrl_torques[1:3]   .= raw_torques[4:6]   # FR
        ctrl_torques[4:6]   .= raw_torques[1:3]   # FL
        ctrl_torques[7:9]   .= raw_torques[10:12] # RR
        ctrl_torques[10:12] .= raw_torques[7:9]   # RL
        d.ctrl .= clamp.(ctrl_torques, -TORQUE_LIMIT, TORQUE_LIMIT)

        U_global[:, 1:end-1] .= U_global[:, 2:end]
        U_global[:, end] .= HOME_POS

        # Query MPPI expert non-invasively on a state copy (does NOT mutate main simulation d)
        state_qpos = vec(copy(d.qpos))
        state_qvel = vec(copy(d.qvel))
        expert_U = query_mppi_expert(m, state_qpos, state_qvel, U_global)

        collect_dataset!(dataset, hist_vec, expert_U)

        if mod(step_in_round, 20) == 0
            println("  DAgger R$current_round Step $step_in_round/$DAGGER_STEPS | height = $(round(d.qpos[3], digits=3)) | vel_x = $(round(d.qvel[1], digits=3)) | dataset = $(length(dataset))")
        end

        qw, qx, qy, qz = d.qpos[4], d.qpos[5], d.qpos[6], d.qpos[7]
        gravity_z = 1.0 - 2.0 * (qx^2 + qy^2)
        if d.qpos[3] < 0.22 || gravity_z < 0.64
            fall_count += 1
            println("  --- DAgger Fall #$fall_count — resetting ---")
            reset_robot!(m, d)
            for t in 1:H; U_global[:, t] .= HOME_POS; end
        end

        if step_in_round == DAGGER_STEPS
            println("\n--- DAgger Round $current_round complete. Retraining diffusion policy on aggregated dataset ---")
            retrain_diffusion_policy!()
            println("=== DAgger Round $current_round retraining done ===\n")

            if current_round == DAGGER_ROUNDS
                policy_ready = true
                println("✓ All DAgger rounds complete! Switching to final pure Diffusion policy control.")
            end
        end

        return
    end

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1–2: Initial MPPI Expert Data Collection (500 Iterations)
    # ═══════════════════════════════════════════════════════════════════
    costs, trajs = mppi_update!(m, d)

    collect_dataset!(dataset, hist_vec, U_global)

    h = d.qpos[3]
    h_diff = h - 0.38
    h_c = h <= 0.38 ? 30000.0 * h_diff^2 : 60000.0 * h_diff^2
    zvel_c = 5000.0 * d.qvel[3]^2
    qw, qx, qy, qz = d.qpos[4], d.qpos[5], d.qpos[6], d.qpos[7]
    gravity_z = 1.0 - 2.0 * (qx^2 + qy^2)
    q_c    = 150000.0 * (1.0 - clamp(gravity_z, -1.0, 1.0))^2
    y_c    = 50000.0 * qz^2
    j_c = 0.0
    margin = 0.15
    for i in 1:12
        qi = d.qpos[7+i]
        lo = JOINT_LOWER[i] + margin; hi = JOINT_UPPER[i] - margin
        if qi < lo; j_c += 5000.0*(lo - qi)^2; elseif qi > hi; j_c += 5000.0*(qi - hi)^2; end
    end
    s_c  = 300.0 * (sum((d.qpos[8:10] .- d.qpos[11:13]).^2) + sum((d.qpos[14:16] .- d.qpos[17:19]).^2))
    vx   = d.qvel[1]
    v_c  = vx <= 0.3 ? 50000.0*(vx-0.3)^2 : 30000.0*(vx-0.3)^2
    la_c = 1500.0   * d.qvel[2]^2
    jv_c = 5.0      * sum(d.qvel[7:18].^2)

    joint_pos = d.qpos[8:19]
    joint_vel = d.qvel[7:18]
    raw_torques = KP .* (U_global[:,1] .- joint_pos) .- KD .* joint_vel
    n_saturated = sum(abs.(raw_torques) .>= TORQUE_LIMIT .- 1e-6)

    push!(log_iter,       iter)
    push!(log_min_cost,   minimum(costs))
    push!(log_height,     d.qpos[3])
    push!(log_vel_x,      d.qvel[1])
    push!(log_height_c,   h_c)
    push!(log_quat_c,     q_c)
    push!(log_joint_c,    j_c)
    push!(log_sym_c,      s_c)
    push!(log_vel_c,      v_c)
    push!(log_lateral_c,  la_c)
    push!(log_jvel_c,     jv_c)
    push!(log_n_sat,      n_saturated)
    push!(log_max_torque, maximum(abs.(raw_torques)))

    if mod(iter, 50) == 0
        println("MPPI Iter $iter / $DATA_ITERS | cost = $(round(minimum(costs), digits=2)) | height = $(round(d.qpos[3], digits=3)) | vel_x = $(round(d.qvel[1], digits=3))")
    end

    # Fall detection
    qw, qx, qy, qz = d.qpos[4], d.qpos[5], d.qpos[6], d.qpos[7]
    gravity_z = 1.0 - 2.0 * (qx^2 + qy^2)

    if d.qpos[3] < 0.22 || gravity_z < 0.64
        fall_count += 1
        println("--- Fall #$fall_count — resetting position ---")
        reset_robot!(m, d)
    end

        # Initial training at the end of initial data collection phase
        if iter == DATA_ITERS
            println("\n✓ MPPI data collection complete ($(DATA_ITERS) iterations)")

            open("metrics.csv", "w") do io
                println(io, "iter,min_cost,height,vel_x,height_cost,quat_cost,joint_cost,sym_cost,vel_cost,lateral_cost,jvel_cost,n_saturated,max_raw_torque")
                for i in eachindex(log_iter)
                    println(io, "$(log_iter[i]),$(log_min_cost[i]),$(log_height[i]),$(log_vel_x[i]),$(log_height_c[i]),$(log_quat_c[i]),$(log_joint_c[i]),$(log_sym_c[i]),$(log_vel_c[i]),$(log_lateral_c[i]),$(log_jvel_c[i]),$(log_n_sat[i]),$(log_max_torque[i])")
                end
            end
            println("✓ Metrics saved to metrics.csv")

            println("\n=== Initial Diffusion Training on $DATA_ITERS MPPI Iterations ===")
            retrain_diffusion_policy!()
            policy_ready = true
            println("=== Starting DAgger correction rounds for Diffusion Policy ===")
        end
    end

init_visualiser()
visualise!(mj_model, mj_data; controller=visual_controller!)