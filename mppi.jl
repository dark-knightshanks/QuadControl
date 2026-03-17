using MuJoCo
using LinearAlgebra
using Random
using Statistics
using Base.Threads


mj_model = load_model("go2/scene.xml")
mj_data  = init_data(mj_model)
include("diffusion1_model.jl")

const K = 50 # Number of samples
#const H = 30    # Horizon length
const λ = 0.2   # Temperature
const Σ = 0.5   # Noise standard deviation
const nx = mj_model.nq + mj_model.nv
#const nu = model.nu
const U_global = zeros(nu, H)

dataset = []

function cost(qpos, qvel, ctrl)
    # Weights
    w_pos = 1000.0    # Position tracking
    w_height = 500.0  # Height tracking
    w_vel = 100.0    # Velocity tracking
    w_ori = 200.0     # Orientation tracking
    w_ang = 20.0      # Angular velocity tracking
    w_ctrl = 0.1      # Control cost

    # Target states
    target_height = 0.45  # Desired standing height
    target_vel_x = 0.6 # Desired forward velocity (m/s)
    target_forward_pos = 0.5
    target_roll = 0.0    # Desired roll angle
    target_pitch = 0.0   # Desired pitch angle

    # Current states
    current_pos = qpos[1:3]
    current_vel = qvel[1:3]
    #current_ori = qpos[7:9] # Roll, pitch, yaw
    #current_ang = qvel[7:9] # Angular velocities
    #CORRECT — extract roll/pitch from quaternion, use qvel[4:6] for angular vel
    qw, qx, qy, qz = qpos[4], qpos[5], qpos[6], qpos[7]
    roll  = atan(2*(qw*qx + qy*qz), 1 - 2*(qx^2 + qy^2))
    pitch = asin(clamp(2*(qw*qy - qz*qx), -1.0, 1.0))
    ori_cost  = w_ori * (roll^2 + pitch^2)

    current_ang = qvel[4:6]   # body angular velocity (free joint)
    ang_cost = w_ang * sum(current_ang .^ 2)

    # Individual cost terms

    # Track desired height
    height_cost = w_height * (current_pos[3] - target_height)^2

    com_cost = 500.0 * (current_pos[1] - target_forward_pos)^2

    # Track desired forward velocity
    vel_cost = w_vel * (current_vel[1] - target_vel_x)^2

    # Keep robot level (minimize roll/pitch)
    ori_cost  = w_ori * (roll^2 + pitch^2)

    # Minimize angular velocities for stability
    current_ang = qvel[4:6]
ang_cost = w_ang * sum(current_ang .^ 2)
    # Penalize sideways motion
    lateral_cost = w_pos * (current_pos[2]^2 + current_vel[2]^2)

    # Penalize control effort
    ctrl_cost = w_ctrl * sum(ctrl .^ 2)

    total_cost = height_cost + vel_cost + ori_cost + ang_cost + lateral_cost + ctrl_cost + com_cost

    return total_cost
end

function rollout(m::Model, d::Data, U::Matrix{Float64}, noise::Array{Float64,3})
    costs = zeros(K)
    trajectories = Array{Float64,3}(undef, K, nu, H)
    @threads for k in 1:K        # running k diff scenarios parallely
        d_copy = init_data(m)
        d_copy.qpos .= d.qpos
        d_copy.qvel .= d.qvel
        cost_sum = 0.0


        for t in 1:H            #  for Horizon
            current_ctrl = vec(U[:, t] + noise[:, t, k]) # add noise 
            d_copy.ctrl .= clamp.(current_ctrl, -20.0, 20.0) # i think so boundiing the control
            mj_step(m, d_copy)  
            println("timestep = \n", t, 
            " time = \n", d_copy.time,
            " pos = \n", d_copy.qpos[1:3],
            " vel = \n", d_copy.qvel[1:3],
            " roll/pitch = \n", d_copy.qpos[7:8])
            cost_sum += cost(d_copy.qpos, d_copy.qvel, d_copy.ctrl)
            trajectories[k, :, t] .= current_ctrl
        end
        costs[k] = cost_sum  
        #println("best cost = ", best_cost)
    end
    best_cost = minimum(costs)
    return costs, trajectories

end


function mppi_update!(m::Model, d::Data)
    noise = randn(nu, H, K) * Σ      
    costs, trajectories = rollout(m, d, U_global, noise)
    print("best cost = ", minimum(costs))
    β = minimum(costs)
    weights = exp.(-1 / λ * (costs .- β))   # setting weights 
    weights ./= sum(weights) + 1e-10           # avg weights 

    for t in 1:H
        weighted_noise = sum(weights[k] * noise[:, t, k] for k in 1:K)
        U_global[:, t] .= clamp.(U_global[:, t] + weighted_noise, -10.0, 10.0)
    end

    d.ctrl .= U_global[:, 1]
    U_global[:, 1:end-1] .= U_global[:, 2:end]
    U_global[:, end] .= 0.0

    return costs, trajectories
end

function collect_dataset!(dataset, mj_data, costs, trajectories; M=10)
    idx = sortperm(costs)
    for i in idx[1:M]
        push!(dataset, (
            Float32.(mj_data.qpos),
            Float32.(mj_data.qvel),
            vec(Float32.(trajectories[i, :, :]))  # flatten to 1D vector
        ))
        
    end
end

# Step 1 — collect data loop (already working )
for iter in 1:50
    costs, trajs = mppi_update!(mj_model, mj_data)
    mj_step(mj_model, mj_data)
    collect_dataset!(dataset, mj_data, costs, trajs; M=10)
    println("Iter $iter | best cost = $(minimum(costs))")
end

# Step 2 — build matrices FIRST
Q = hcat([vcat(d[1], d[2]) for d in dataset]...)
U = hcat([d[3] for d in dataset]...)

# Step 3 — compute stats SECOND
μ_q = mean(Q, dims=2)
σ_q = std(Q, dims=2) .+ 1e-6

μ_u = mean(U, dims=2)
σ_u = std(U, dims=2) .+ 1e-6

# Step 4 — prepare dataset THIRD (needs μ_q, σ_q, μ_u, σ_u)
X_state, X_ctrl = prepare_dataset(dataset, μ_q, σ_q, μ_u, σ_u)

# Step 5 — train LAST
train_diffusion!(model, X_state, X_ctrl; epochs=100)


init_visualiser()
visualise!(mj_model, mj_data; controller=mppi_update!)