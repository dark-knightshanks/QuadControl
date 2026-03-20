using Flux
using LinearAlgebra
using Statistics
using Random
using Flux: Dense, Chain, relu


const lr = 3e-4
const T_diff = 100    # Horizon length
const nu = mj_model.nu      # ← relies on mj_model from mppi.jl
const H = 40
state_dim = mj_model.nq + mj_model.nv
traj_dim  = nu * H
in_dim    = traj_dim + state_dim + 1
hidden_dim = 512
hidd_dim = 256

# trajectories: K x nu x H
# Convert to a Flux-friendly format: (samples, features)
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



model = Chain(
    Dense(in_dim, hidden_dim, relu),
    Dense(hidden_dim, hidden_dim, relu),
    Dense(hidden_dim, hidd_dim, relu),
    Dense(hidd_dim, traj_dim)
)



function q_sample(x0, t, α)
    ϵ = randn(Float32, size(x0))
    xt = sqrt(α[t]) .* x0 .+ sqrt(1 - α[t]) .* ϵ
    return xt, ϵ
end

function train_diffusion!(model, X_state, X_ctrl; epochs, lr=1e-4)
    opt = Flux.setup(Adam(lr), model)
    T = 100
    β_sched = collect(LinRange(1e-4, 0.02, T))
    α = cumprod(1.0f0 .- β_sched)

    N = size(X_ctrl, 2)

    for epoch in 1:epochs
        total_loss = 0.0

        for i in 1:N
            u0 = Float32.(X_ctrl[:, i])   # force Float32
            s  = Float32.(X_state[:, i])  #  force Float32
        
            t = rand(1:T)
            xt, ϵ = q_sample(u0, t, α)
        
            t_embed = Float32(t / T)
            input = Float32.(vcat(xt, s, t_embed))  #  ensure whole input is Float32
        
            loss, grads = Flux.withgradient(model) do m
                ϵ̂ = m(input)
                mean((ϵ̂ .- ϵ).^2)
            end
        
            Flux.update!(opt, model, grads[1])  # grads[1] not grads
            total_loss += loss
        end
        println("Epoch $epoch | loss = $(total_loss / N)")
    end
end



function ddim_sample(model, state, α, T_diff)
    x_t = randn(Float32, traj_dim)

    for t in T_diff:-1:1
        t_embed = fill(Float32(t/T_diff), 1)
        input = vcat(x_t, state, t_embed)

        ϵ_pred = model(input)

        a_t = α[t]
        a_prev = t > 1 ? α[t-1] : 1.0f0

        x_t = sqrt(a_prev) .* ((x_t .- sqrt(1 - a_t) .* ϵ_pred) ./ sqrt(a_t)) +
              sqrt(1 - a_prev) .* ϵ_pred
    end

    return reshape(x_t, nu, H)
end
