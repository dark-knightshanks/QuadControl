using MuJoCo
using LinearAlgebra
using Random
using Statistics
using Base.Threads

# ── Load Model ────────────────────────────────────────────────────────
mj_model = load_model("go2/scene.xml")
mj_data  = init_data(mj_model)
include("mlp_policy.jl")

const nu = mj_model.nu
const H  = 40
const K  = 150   # Full MPPI budget for high-quality expert data collection
const λ  = 0.1
const KP = 80.0
const KD = 4.0

# Go2 stable standing pose
const HOME_POS = Float64[
    0.0, 0.9, -1.8,   # FL
    0.0, 0.9, -1.8,   # FR
    0.0, 0.9, -1.8,   # RL
    0.0, 0.9, -1.8    # RR
]

const noise_sigma = Float64[
    0.08, 0.15, 0.15,
    0.08, 0.15, 0.15,
    0.08, 0.15, 0.15,
    0.08, 0.15, 0.15
]

const TORQUE_LIMIT = Float64[
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43,
    23.7, 23.7, 45.43
]

const JOINT_LOWER = Float64[
    -1.0472, -1.5708, -2.7227,
    -1.0472, -1.5708, -2.7227,
    -1.0472, -0.5236, -2.7227,
    -1.0472, -0.5236, -2.7227
]
const JOINT_UPPER = Float64[
     1.0472,  3.4907, -0.83776,
     1.0472,  3.4907, -0.83776,
     1.0472,  4.5379, -0.83776,
     1.0472,  4.5379, -0.83776
]

const U_global = zeros(nu, H)
dataset = []

# ── Cost Function ─────────────────────────────────────────────────────
function cost(qpos, qvel, pos_target)
    target_height = 0.38
    target_vel_x  = 0.3

    if qpos[3] < 0.20
        return 1_000_000.0
    end

    h = qpos[3]
    h_diff = h - target_height
    height_cost = h <= target_height ? 30000.0 * h_diff^2 : 60000.0 * h_diff^2
    z_vel_cost = 5000.0 * qvel[3]^2

    qw, qx, qy, qz = qpos[4], qpos[5], qpos[6], qpos[7]
    gravity_z = 1.0 - 2.0 * (qx^2 + qy^2)
    quat_cost = 150000.0 * (1.0 - clamp(gravity_z, -1.0, 1.0))^2
    yaw_cost = 50000.0 * qz^2

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

    # Strong leg symmetry cost (prevents one-legged standing / splayed rear legs)
    sym_cost = 5000.0 * (
        sum((qpos[8:10]  .- qpos[11:13]).^2) +
        sum((qpos[14:16] .- qpos[17:19]).^2)
    )

    # Base stance regularizer (keeps joint angles centered near natural standing posture)
    pose_reg_cost = 2000.0 * sum((qpos[8:19] .- HOME_POS).^2)

    vel_x = qvel[1]
    vel_cost = if vel_x <= target_vel_x
        50000.0 * (vel_x - target_vel_x)^2        # Strong forward pull towards 0.3 m/s
    else
        120000.0 * (vel_x - target_vel_x)^2       # Heavy penalty for overspeeding >0.3 m/s to prevent forward tilt
    end
    lateral_cost = 1500.0 * qvel[2]^2
    jvel_cost = 5.0 * sum(qvel[7:18].^2)

    return height_cost + z_vel_cost + quat_cost + yaw_cost + joint_range_cost + sym_cost + pose_reg_cost + vel_cost + lateral_cost + jvel_cost
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
            pos_target = vec(U[:, t] + noise[:, t, k])
            joint_pos = d_copy.qpos[8:19]
            joint_vel = d_copy.qvel[7:18]
            raw_torques = KP .* (pos_target .- joint_pos) .- KD .* joint_vel

            ctrl_torques = zeros(12)
            ctrl_torques[1:3]   .= raw_torques[4:6]
            ctrl_torques[4:6]   .= raw_torques[1:3]
            ctrl_torques[7:9]   .= raw_torques[10:12]
            ctrl_torques[10:12] .= raw_torques[7:9]

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
    λ_eff = 0.15 * cost_std
    weights = exp.(-1 / λ_eff * (costs .- β))
    weights ./= sum(weights) + 1e-10

    for t in 1:H
        weighted_noise = sum(weights[k] * noise[:, t, k] for k in 1:K)
        U_global[:, t] .= clamp.(U_global[:, t] + weighted_noise, JOINT_LOWER, JOINT_UPPER)
    end

    joint_pos = d.qpos[8:19]
    joint_vel = d.qvel[7:18]
    raw_torques = KP .* (U_global[:, 1] .- joint_pos) .- KD .* joint_vel

    ctrl_torques = zeros(12)
    ctrl_torques[1:3]   .= raw_torques[4:6]
    ctrl_torques[4:6]   .= raw_torques[1:3]
    ctrl_torques[7:9]   .= raw_torques[10:12]
    ctrl_torques[10:12] .= raw_torques[7:9]

    d.ctrl .= clamp.(ctrl_torques, -TORQUE_LIMIT, TORQUE_LIMIT)

    U_global[:, 1:end-1] .= U_global[:, 2:end]
    U_global[:, end] .= HOME_POS

    return costs, trajectories
end

# ── State History Queue ──────────────────────────────────────────────────
const state_history = Vector{Float32}[]

function get_history_vector(current_state::AbstractVector{Float32})
    curr_vec = vec(current_state)
    push!(state_history, curr_vec)
    while length(state_history) > bc_history_len
        popfirst!(state_history)
    end
    while length(state_history) < bc_history_len
        pushfirst!(state_history, state_history[1])
    end
    return vcat(state_history...)
end

function clear_history_buffer!(initial_state::AbstractVector{Float32})
    empty!(state_history)
    init_vec = vec(initial_state)
    for _ in 1:bc_history_len
        push!(state_history, init_vec)
    end
end

# ── Dataset Collection (History Vector -> Residual Target) ───────────────
function collect_dataset!(dataset, history_vec::Vector{Float32}, U_opt::Matrix{Float64})
    # Target is residual relative to HOME_POS baseline stance
    target_residual = U_opt .- HOME_POS
    push!(dataset, (
        history_vec,
        vec(Float32.(target_residual))
    ))
end

# ── Reset ─────────────────────────────────────────────────────────────
function reset_robot!(m, d)
    MuJoCo.mj_resetData(m, d)
    d.qpos[1:3]  .= [0.0, 0.0, 0.27]
    d.qpos[4:7]  .= [1.0, 0.0, 0.0, 0.0]
    d.qpos[8:19] .= HOME_POS
    d.qvel       .= 0.0
    for _ in 1:10
        raw_torques = KP .* (HOME_POS .- d.qpos[8:19]) .- KD .* d.qvel[7:18]
        ctrl_torques = zeros(12)
        ctrl_torques[1:3]   .= raw_torques[4:6]
        ctrl_torques[4:6]   .= raw_torques[1:3]
        ctrl_torques[7:9]   .= raw_torques[10:12]
        ctrl_torques[10:12] .= raw_torques[7:9]
        d.ctrl .= clamp.(ctrl_torques, -TORQUE_LIMIT, TORQUE_LIMIT)
        mj_step(m, d)
    end
    for t in 1:H; U_global[:, t] .= HOME_POS; end
    clear_history_buffer!(vec(Float32.(vcat(d.qpos, d.qvel))))
end

# ══════════════════════════════════════════════════════════════════════
# CONTROLLER
# ══════════════════════════════════════════════════════════════════════
reset_robot!(mj_model, mj_data)
for t in 1:H; U_global[:, t] .= HOME_POS; end
global fall_count = 0
global vis_iter = 0

const DATA_ITERS   = 500     # Collect 500 MPPI expert iterations
const POLICY_ITERS = 500     # Run MLP policy for 500 iterations
const EXEC_STEPS   = 4       # Receding horizon: re-predict every 4 steps

# DAgger configuration
const DAGGER_ROUNDS   = 2     # Number of DAgger correction rounds
const DAGGER_STEPS    = 200   # Steps per DAgger round (MLP acts, MPPI labels)
const FINAL_MLP_ITERS = 500   # Final pure MLP execution after all DAgger rounds

# Globals
global policy_ready = false
global dagger_round = 0
global dagger_step_in_round = 0
global p4_μ_q = nothing
global p4_σ_q = nothing
global p4_μ_u = nothing
global p4_σ_u = nothing

function compute_phase_boundary()
    # Phase 1 ends at DATA_ITERS
    # Each DAgger round is DAGGER_STEPS long + retrain at last step
    dagger_total = DAGGER_ROUNDS * DAGGER_STEPS
    # Final MLP execution starts after all DAgger rounds
    return DATA_ITERS + dagger_total
end

function retrain_policy!()
    global p4_μ_q, p4_σ_q, p4_μ_u, p4_σ_u

    # Filter dataset: drop samples where current state height < 0.28m
    filtered = filter(d -> d[1][3] >= 0.28, dataset)
    println("✓ Retraining on history dataset: $(length(filtered)) / $(length(dataset)) clean samples")

    Q = hcat([d[1] for d in filtered]...)
    U_ds = hcat([d[2] for d in filtered]...)
    p4_μ_q = Float32.(mean(Q, dims=2))
    p4_σ_q = Float32.(std(Q, dims=2) .+ 1e-6)
    p4_μ_u = Float32.(mean(U_ds, dims=2))
    p4_σ_u = Float32.(std(U_ds, dims=2) .+ 1e-6)

    X_state, Y_traj = bc_prepare_dataset(filtered, p4_μ_q, p4_σ_q, p4_μ_u, p4_σ_u)
    train_bc_policy!(bc_policy, X_state, Y_traj; epochs=120, batchsize=32, lr=5e-4)
end

function visual_controller!(m, d)
    global vis_iter, fall_count, policy_ready
    global dagger_round, dagger_step_in_round
    global p4_μ_q, p4_σ_q, p4_μ_u, p4_σ_u

    vis_iter += 1
    iter = vis_iter

    dagger_end = compute_phase_boundary()
    total_end  = dagger_end + FINAL_MLP_ITERS

    # Update state history buffer with current frame
    curr_state_vec = vec(Float32.(vcat(d.qpos, d.qvel)))
    hist_vec = get_history_vector(curr_state_vec)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: Final Pure MLP Policy (History-Conditioned + Residual)
    # ═══════════════════════════════════════════════════════════════════
    if iter > dagger_end
        if iter > total_end
            return
        end

        final_step = iter - dagger_end

        if policy_ready && mod(final_step - 1, EXEC_STEPS) == 0
            traj = bc_predict(bc_policy, hist_vec, p4_μ_q, p4_σ_q, p4_μ_u, p4_σ_u)
            global U_global .= clamp.(Float64.(traj), JOINT_LOWER, JOINT_UPPER)
        end

        # PD control
        joint_pos = d.qpos[8:19]
        joint_vel = d.qvel[7:18]
        raw_torques = KP .* (U_global[:, 1] .- joint_pos) .- KD .* joint_vel
        ctrl_torques = zeros(12)
        ctrl_torques[1:3]   .= raw_torques[4:6]
        ctrl_torques[4:6]   .= raw_torques[1:3]
        ctrl_torques[7:9]   .= raw_torques[10:12]
        ctrl_torques[10:12] .= raw_torques[7:9]
        d.ctrl .= clamp.(ctrl_torques, -TORQUE_LIMIT, TORQUE_LIMIT)

        U_global[:, 1:end-1] .= U_global[:, 2:end]
        U_global[:, end] .= HOME_POS

        if mod(final_step, 10) == 0
            println("Final MLP Step $final_step | height = $(round(d.qpos[3], digits=3)) | vel_x = $(round(d.qvel[1], digits=3))")
        end

        # Fall detection
        qw, qx, qy, qz = d.qpos[4], d.qpos[5], d.qpos[6], d.qpos[7]
        gravity_z = 1.0 - 2.0 * (qx^2 + qy^2)
        if d.qpos[3] < 0.22 || gravity_z < 0.64
            fall_count += 1
            println("--- Final MLP Fall #$fall_count — resetting ---")
            reset_robot!(m, d)
        end

        return
    end

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: DAgger Rounds
    # ═══════════════════════════════════════════════════════════════════
    if iter > DATA_ITERS && iter <= dagger_end
        dagger_offset = iter - DATA_ITERS
        current_round = div(dagger_offset - 1, DAGGER_STEPS) + 1
        step_in_round = mod(dagger_offset - 1, DAGGER_STEPS) + 1

        if step_in_round == 1 && current_round != dagger_round
            dagger_round = current_round
            println("\n=== DAgger Round $dagger_round / $DAGGER_ROUNDS: MLP acts, MPPI corrects ($DAGGER_STEPS steps) ===")
            reset_robot!(m, d)
            for t in 1:H; U_global[:, t] .= HOME_POS; end
        end

        if policy_ready && mod(step_in_round - 1, EXEC_STEPS) == 0
            traj = bc_predict(bc_policy, hist_vec, p4_μ_q, p4_σ_q, p4_μ_u, p4_σ_u)
            global U_global .= clamp.(Float64.(traj), JOINT_LOWER, JOINT_UPPER)
        end

        joint_pos = d.qpos[8:19]
        joint_vel = d.qvel[7:18]
        raw_torques = KP .* (U_global[:, 1] .- joint_pos) .- KD .* joint_vel
        ctrl_torques = zeros(12)
        ctrl_torques[1:3]   .= raw_torques[4:6]
        ctrl_torques[4:6]   .= raw_torques[1:3]
        ctrl_torques[7:9]   .= raw_torques[10:12]
        ctrl_torques[10:12] .= raw_torques[7:9]
        d.ctrl .= clamp.(ctrl_torques, -TORQUE_LIMIT, TORQUE_LIMIT)

        U_global[:, 1:end-1] .= U_global[:, 2:end]
        U_global[:, end] .= HOME_POS

        # Query MPPI expert
        saved_U = copy(U_global)
        for t in 1:H; U_global[:, t] .= 0.7 .* saved_U[:, t] .+ 0.3 .* HOME_POS; end
        for _ in 1:3
            mppi_update!(m, d)
        end
        expert_U = copy(U_global)
        U_global .= saved_U

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
            println("\n--- DAgger Round $current_round complete. Retraining on aggregated data ---")
            retrain_policy!()
            println("=== DAgger Round $current_round retraining done ===\n")

            if current_round == DAGGER_ROUNDS
                policy_ready = true
                println("✓ All DAgger rounds complete! Switching to final pure MLP policy control.")
            end
        end

        return
    end

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1-2: MPPI Expert Data Collection + Initial Training
    # ═══════════════════════════════════════════════════════════════════
    costs, trajs = mppi_update!(m, d)
    collect_dataset!(dataset, hist_vec, U_global)

    if mod(iter, 50) == 0
        println("MPPI Iter $iter / $DATA_ITERS | cost = $(round(minimum(costs), digits=2)) | height = $(round(d.qpos[3], digits=3)) | vel_x = $(round(d.qvel[1], digits=3))")
    end

    # Fall detection
    qw, qx, qy, qz = d.qpos[4], d.qpos[5], d.qpos[6], d.qpos[7]
    gravity_z = 1.0 - 2.0 * (qx^2 + qy^2)
    if d.qpos[3] < 0.22 || gravity_z < 0.64
        fall_count += 1
        println("--- Fall #$fall_count — resetting ---")
        reset_robot!(m, d)
    end

    # Phase 2: Initial MLP training at end of data collection
    if iter == DATA_ITERS
        println("\n✓ MPPI data collection complete ($DATA_ITERS iterations, $(length(dataset)) samples)")
        retrain_policy!()
        policy_ready = true
        println("=== Starting DAgger correction rounds ===")
    end
end

init_visualiser()
visualise!(mj_model, mj_data; controller=visual_controller!)
