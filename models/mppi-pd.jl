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
        30000.0 * (vel_x - target_vel_x)^2        # Penalize overshooting 0.3 m/s
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

# ── Dataset (Saves true converged U_global trajectory) ─────────────────
function collect_dataset!(dataset, mj_data, U_opt::Matrix{Float64})
    push!(dataset, (
        Float32.(mj_data.qpos),
        Float32.(mj_data.qvel),
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

const MAX_ITERS = 300
const PHASE4_ITERS = 200   # Run 200 more iterations under diffusion-warm-started MPPI

# Globals for Phase 4 normalization stats (set during training)
global phase4_ready = false
global phase4_μ_q = nothing
global phase4_σ_q = nothing
global phase4_μ_u = nothing
global phase4_σ_u = nothing
global phase4_α   = nothing

function visual_controller!(m, d)
    global vis_iter, fall_count, phase4_ready
    global phase4_μ_q, phase4_σ_q, phase4_μ_u, phase4_σ_u, phase4_α

    vis_iter += 1
    iter = vis_iter

    # ── Phase 4: Pure Diffusion Policy (Receding Horizon) ─────────────
    #   No MPPI — diffusion model is the sole controller.
    #   Re-predict trajectory every EXEC_STEPS from current state.
    if iter > MAX_ITERS
        if iter > MAX_ITERS + PHASE4_ITERS
            return
        end

        EXEC_STEPS = 4  # Execute only 4 steps before re-planning

        phase4_step = iter - MAX_ITERS  # 1-indexed phase4 counter

        # Re-predict trajectory from current state every EXEC_STEPS
        if phase4_ready && mod(phase4_step - 1, EXEC_STEPS) == 0
            current_state = Float32.(vcat(d.qpos, d.qvel))
            norm_state = (current_state .- phase4_μ_q) ./ phase4_σ_q
            u_ws = ddim_sample(model, norm_state, phase4_α, T_diff; num_steps=20)
            μ_u_mat = reshape(phase4_μ_u, nu, H)
            σ_u_mat = reshape(phase4_σ_u, nu, H)
            global U_global .= clamp.((u_ws .* σ_u_mat) .+ μ_u_mat, JOINT_LOWER, JOINT_UPPER)
        end

        # Apply PD control from U_global[:, 1] (pure diffusion, no MPPI)
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

        if mod(phase4_step, 10) == 0
            println("Phase4 [Diffusion Only] Step $phase4_step | height = $(round(d.qpos[3], digits=3)) | vel_x = $(round(d.qvel[1], digits=3))")
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

    # ── Phase 1–2: Data Collection with MPPI ────────────────────────
    # Record state BEFORE MPPI step
    state_qpos = copy(d.qpos)
    state_qvel = copy(d.qvel)

    costs, trajs = mppi_update!(m, d)

    # Store true converged U_global paired with exact state
    collect_dataset!(dataset, (qpos=state_qpos, qvel=state_qvel), U_global)

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

    println("Iter $iter | cost = $(round(minimum(costs), digits=2)) | height = $(round(d.qpos[3], digits=3)) | vel_x = $(round(d.qvel[1], digits=3))")
    println("    breakdown | height=$(round(h_c,digits=1)) quat=$(round(q_c,digits=1)) jrange=$(round(j_c,digits=1)) sym=$(round(s_c,digits=1)) vel=$(round(v_c,digits=1)) lateral=$(round(la_c,digits=1)) jvel=$(round(jv_c,digits=1))")

    # Fall detection
    qw, qx, qy, qz = d.qpos[4], d.qpos[5], d.qpos[6], d.qpos[7]
    gravity_z = 1.0 - 2.0 * (qx^2 + qy^2)

    if d.qpos[3] < 0.22 || gravity_z < 0.64
        fall_count += 1
        println("--- Fall #$fall_count (height=$(round(d.qpos[3],digits=2)), gravity_z=$(round(gravity_z,digits=2))) — resetting position ---")
        reset_robot!(m, d)
    end

    # ── Phase 3: Train + Phase 4: Initialize ────────────────────────
    if iter == MAX_ITERS
        println("\n✓ Data collection complete ($(MAX_ITERS) iterations)")

        open("metrics.csv", "w") do io
            println(io, "iter,min_cost,height,vel_x,height_cost,quat_cost,joint_cost,sym_cost,vel_cost,lateral_cost,jvel_cost,n_saturated,max_raw_torque")
            for i in eachindex(log_iter)
                println(io, "$(log_iter[i]),$(log_min_cost[i]),$(log_height[i]),$(log_vel_x[i]),$(log_height_c[i]),$(log_quat_c[i]),$(log_joint_c[i]),$(log_sym_c[i]),$(log_vel_c[i]),$(log_lateral_c[i]),$(log_jvel_c[i]),$(log_n_sat[i]),$(log_max_torque[i])")
            end
        end
        println("✓ Metrics saved to metrics.csv")

        # Filter out fallen states (height < 0.28m)
        filtered_dataset = filter(d -> d[1][3] >= 0.28, dataset)
        println("✓ Dataset collected: $(length(dataset)) steps. Clean non-fallen samples: $(length(filtered_dataset))")

        Q = hcat([vcat(d[1], d[2]) for d in filtered_dataset]...)
        U_ds = hcat([d[3] for d in filtered_dataset]...)
        phase4_μ_q = Float32.(mean(Q, dims=2))
        phase4_σ_q = Float32.(std(Q,  dims=2) .+ 1e-6)
        phase4_μ_u = Float32.(mean(U_ds, dims=2))
        phase4_σ_u = Float32.(std(U_ds,  dims=2) .+ 1e-6)
        X_state, X_ctrl = prepare_dataset(filtered_dataset, phase4_μ_q, phase4_σ_q, phase4_μ_u, phase4_σ_u)

        # Phase 3: Train Diffusion Model on True MPPI Trajectories
        train_diffusion!(model, X_state, X_ctrl; epochs=100, batchsize=64, lr=1e-3)

        # Precompute noise schedule for Phase 4
        β_sched = Float32.(collect(LinRange(1e-4, 0.18, T_diff)))
        phase4_α = cumprod(1.0f0 .- β_sched)

        # Initial DDIM warm-start
        println("=== Phase 4: Initializing DDIM Warm-Start ===")
        current_state = Float32.(vcat(d.qpos, d.qvel))
        norm_state = (current_state .- phase4_μ_q) ./ phase4_σ_q
        u_ws = ddim_sample(model, norm_state, phase4_α, T_diff; num_steps=20)
        μ_u_mat = reshape(phase4_μ_u, nu, H)
        σ_u_mat = reshape(phase4_σ_u, nu, H)
        global U_global .= (u_ws .* σ_u_mat) .+ μ_u_mat

        phase4_ready = true
        println("✓ Phase 4 active: Diffusion warm-start + MPPI refinement running for $PHASE4_ITERS iterations")
    end
end

init_visualiser()
visualise!(mj_model, mj_data; controller=visual_controller!)