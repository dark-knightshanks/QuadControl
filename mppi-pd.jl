using MuJoCo
using LinearAlgebra
using Random
using Statistics
using Base.Threads

mj_model = load_model("go2/scene.xml")
mj_data  = init_data(mj_model)
include("diffusion1_model.jl")

const K  = 30
const λ  = 0.1
const KP = 50.0    # PD proportional gain
const KD = 3.0     # PD derivative gain

const noise_sigma = Float64[
    0.15, 0.25, 0.25,   # increased — was 0.06, 0.1, 0.1
    0.15, 0.25, 0.25,
    0.15, 0.25, 0.25,
    0.15, 0.25, 0.25
]
# U_global now holds JOINT POSITION TARGETS not torques
const U_global = zeros(nu, H)
dataset = []

# ── Cost ─────────────────────────────────────────────────────────────
function cost(qpos, qvel, pos_target)
    target_height = 0.445
    target_vel_x  = 0.3
    target_quat   = [1.0, 0.0, 0.0, 0.0]

    if qpos[3] < 0.20
        return 1_000_000.0
    end

    # Height
    height_cost = 5000.0 * (qpos[3] - target_height)^2

    # Orientation via quaternion distance
    q = [qpos[4], qpos[5], qpos[6], qpos[7]]
    quat_dist = 1.0 - clamp(abs(sum(q .* target_quat)), 0.0, 1.0)
    quat_cost = 400000.0 * quat_dist^2

    # Joint tracking — zeros is the natural standing pose
    joint_cost = 800.0 * sum((qpos[8:19] .- zeros(12)).^2)

    # Forward velocity — penalize backward motion harder
    vel_x = qvel[1]
    vel_cost = vel_x >= 0 ?
        500.0 * (vel_x - target_vel_x)^2 :
        8000.0 * vel_x^2
    forward_bonus = -500.0 * max(0.0, qvel[1])
    # Lateral drift
    lateral_cost = 500.0 * qvel[2]^2

    # Joint velocity smoothness
    jvel_cost = 0.1 * sum(qvel[7:18].^2)

    return height_cost + quat_cost + joint_cost + vel_cost + lateral_cost + jvel_cost + forward_bonus
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

            # PD converts position targets to torques
            joint_pos = d_copy.qpos[8:19]
            joint_vel = d_copy.qvel[7:18]
            torques = KP .* (pos_target .- joint_pos) .- KD .* joint_vel

            d_copy.ctrl .= clamp.(torques, -33.5, 33.5)
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
    weights = exp.(-1 / λ * (costs .- β))
    weights ./= sum(weights) + 1e-10

    for t in 1:H
        weighted_noise = sum(weights[k] * noise[:, t, k] for k in 1:K)
        U_global[:, t] .= clamp.(U_global[:, t] + weighted_noise, -π, π)
    end

    # Apply via PD to real robot
    joint_pos = d.qpos[8:19]
    joint_vel = d.qvel[7:18]
    torques = KP .* (U_global[:, 1] .- joint_pos) .- KD .* joint_vel
    d.ctrl .= clamp.(torques, -33.5, 33.5)

    U_global[:, 1:end-1] .= U_global[:, 2:end]
    U_global[:, end] .= 0.0

    return costs, trajectories
end

# ── Dataset ───────────────────────────────────────────────────────────
function collect_dataset!(dataset, mj_data, costs, trajectories; M=10)
    idx = sortperm(costs)
    for i in idx[1:M]
        push!(dataset, (
            Float32.(mj_data.qpos),
            Float32.(mj_data.qvel),
            vec(Float32.(trajectories[i, :, :]))
        ))
    end
end

# ── Reset ─────────────────────────────────────────────────────────────
function reset_robot!(m, d)
    MuJoCo.mj_resetData(m, d)
    d.qpos[3]    = 0.445
    d.qpos[4]    = 1.0
    d.qpos[5]    = 0.0
    d.qpos[6]    = 0.0
    d.qpos[7]    = 0.0
    d.qpos[8:19] .= 0.0
    d.qvel       .= 0.0
    MuJoCo.mj_forward(m, d)
end

# ── Phase 1: Data Collection ──────────────────────────────────────────
reset_robot!(mj_model, mj_data)
U_global .= 0.0
global fall_count = 0

for iter in 1:300
    costs, trajs = mppi_update!(mj_model, mj_data)
    mj_step(mj_model, mj_data)
    collect_dataset!(dataset, mj_data, costs, trajs; M=10)
    println("Iter $iter | cost = $(round(minimum(costs), digits=2)) | height = $(round(mj_data.qpos[3], digits=3)) | vel_x = $(round(mj_data.qvel[1], digits=3))")

    if mj_data.qpos[3] < 0.20
        global fall_count += 1
        println("--- Fall #$fall_count — resetting position, keeping controls ---")
        reset_robot!(mj_model, mj_data)
    end
end

# ── Phase 2: Build Dataset ────────────────────────────────────────────
Q = hcat([vcat(d[1], d[2]) for d in dataset]...)
U = hcat([d[3] for d in dataset]...)

μ_q = Float32.(mean(Q, dims=2))
σ_q = Float32.(std(Q,  dims=2) .+ 1e-6)
μ_u = Float32.(mean(U, dims=2))
σ_u = Float32.(std(U,  dims=2) .+ 1e-6)

X_state, X_ctrl = prepare_dataset(dataset, μ_q, σ_q, μ_u, σ_u)

# ── Phase 3: Train Diffusion ──────────────────────────────────────────
train_diffusion!(model, X_state, X_ctrl; epochs=100)

# ── Phase 4: Visualise ────────────────────────────────────────────────
reset_robot!(mj_model, mj_data)
U_global .= 0.0
init_visualiser()
visualise!(mj_model, mj_data; controller=mppi_update!)