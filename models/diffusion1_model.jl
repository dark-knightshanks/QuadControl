using Flux
using LinearAlgebra
using Statistics
using Random
using Flux: Dense, Chain, relu, DataLoader

const lr = 1e-3
const T_diff = 50     # Diffusion horizon length
const nu = mj_model.nu # relies on mj_model
const H = 40
state_dim = mj_model.nq + mj_model.nv
traj_dim  = nu * H
const t_dim = 16       # 16-dimensional Sinusoidal time embedding
in_dim    = traj_dim + state_dim + t_dim
hidden_dim = 512

function prepare_dataset(dataset, μ_q, σ_q, μ_u, σ_u)
    X_state = Vector{Float32}[]
    X_ctrl  = Vector{Float32}[]

    for (qpos, qvel, traj) in dataset
        q = vcat(qpos, qvel)
        qn = vec((q .- μ_q) ./ σ_q)
        un = vec((traj .- μ_u) ./ σ_u)

        push!(X_state, Float32.(qn))
        push!(X_ctrl,  Float32.(un))
    end

    return hcat(X_state...), hcat(X_ctrl...)
end

# 1. Sinusoidal Time Embedding: converts scalar t -> 16D frequency vector
function sinusoidal_embedding(t::Int, T::Int, dim::Int=16)
    half_dim = div(dim, 2)
    emb = log(10000.0f0) / (half_dim - 1)
    frequencies = exp.(-emb .* (0:half_dim-1))
    t_val = Float32(t)
    args = t_val .* frequencies
    return Float32.(vcat(sin.(args), cos.(args)))
end

function batch_sinusoidal_embedding(t_vec::Vector{Int}, T::Int, dim::Int=16)
    B = length(t_vec)
    emb_mat = zeros(Float32, dim, B)
    for i in 1:B
        emb_mat[:, i] = sinusoidal_embedding(t_vec[i], T, dim)
    end
    return emb_mat
end

model = Chain(
    Dense(in_dim, hidden_dim, relu),
    Dense(hidden_dim, hidden_dim, relu),
    Dense(hidden_dim, hidden_dim, relu),
    Dense(hidden_dim, traj_dim)  # Direct x0 prediction
)

# Forward Diffusion Process: q(x_t | x_0) = N(x_t; sqrt(α_bar_t)*x_0, (1 - α_bar_t)*I)
function q_sample(x0, t, α)
    ϵ = randn(Float32, size(x0))
    xt = sqrt(α[t]) .* x0 .+ sqrt(1.0f0 - α[t]) .* ϵ
    return xt
end

# Vectorized Diffusion Training with x0-Loss Optimization
function train_diffusion!(model, X_state, X_ctrl; epochs=80, batchsize=64, lr=1e-3)
    opt = Flux.setup(Adam(lr), model)
    T = T_diff
    β_sched = Float32.(collect(LinRange(1e-4, 0.18, T)))
    α = cumprod(1.0f0 .- β_sched)

    loader = DataLoader((X_ctrl, X_state), batchsize=batchsize, shuffle=true)

    println("\n=== Starting x0-Prediction Diffusion Training ($epochs Epochs, Batch Size $batchsize) ===")

    for epoch in 1:epochs
        total_loss = 0.0f0
        num_batches = 0

        for (u0_b, s_b) in loader
            B = size(u0_b, 2)
            t_vec = rand(1:T, B)

            xt_b = zeros(Float32, size(u0_b))
            for i in 1:B
                t = t_vec[i]
                xt_b[:, i] = q_sample(u0_b[:, i], t, α)
            end

            t_embed_b = batch_sinusoidal_embedding(t_vec, T, t_dim)
            input_b   = vcat(xt_b, s_b, t_embed_b)

            # Direct x0 Loss: || m(x_t, s, t) - u_0 ||^2
            loss, grads = Flux.withgradient(model) do m
                x0_pred = m(input_b)
                mean((x0_pred .- u0_b).^2)
            end

            Flux.update!(opt, model, grads[1])
            total_loss += loss
            num_batches += 1
        end

        avg_loss = total_loss / num_batches
        if epoch % 5 == 0 || epoch == 1 || epoch == epochs
            println("Epoch $(lpad(epoch, 2)) / $epochs | MSE x0 Loss = $(round(avg_loss, digits=6))")
        end
    end
    println("✓ Diffusion Model Training Complete!\n")
end

# Accelerated DDIM Sampler using predicted x0
function ddim_sample(model, state, α, T_diff; num_steps=20)
    x_t = randn(Float32, traj_dim)
    step_stride = max(1, div(T_diff, num_steps))
    timesteps = T_diff:-step_stride:1

    for idx in 1:length(timesteps)
        t = timesteps[idx]
        t_embed = sinusoidal_embedding(t, T_diff, t_dim)
        input = vcat(x_t, Float32.(state), t_embed)

        # Direct prediction of x0 trajectory from noisy x_t
        pred_x0 = model(input)

        a_t = α[t]
        prev_idx = idx + 1
        a_prev = prev_idx <= length(timesteps) ? α[timesteps[prev_idx]] : 1.0f0

        # Derived noise prediction: ϵ_pred = (x_t - sqrt(a_t)*pred_x0) / sqrt(1 - a_t)
        ϵ_pred = (x_t .- sqrt(a_t) .* pred_x0) ./ sqrt(1.0f0 - a_t)

        # Deterministic DDIM trajectory update: x_{t-1} = sqrt(a_prev)*pred_x0 + sqrt(1 - a_prev)*ϵ_pred
        x_t = sqrt(a_prev) .* pred_x0 .+ sqrt(1.0f0 - a_prev) .* ϵ_pred
    end

    return reshape(x_t, nu, H)
end
