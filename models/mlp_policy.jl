using Flux
using LinearAlgebra
using Statistics
using Random
using Flux: Dense, Chain, relu, DataLoader

# ── Dimensions ─────────────────────────────────────────────────────────
const bc_nu = mj_model.nu        # 12 joints
const bc_H  = 40                 # planning horizon
const bc_single_state_dim = mj_model.nq + mj_model.nv  # 37
const bc_history_len = 4
const bc_state_dim = bc_single_state_dim * bc_history_len  # 148 input dims
const bc_traj_dim  = bc_nu * bc_H                         # 480 output dims

# ── MLP Policy Network (History-Conditioned Residual Target) ─────────────
bc_policy = Chain(
    Dense(bc_state_dim, 512, relu),
    Dense(512, 512, relu),
    Dense(512, 512, relu),
    Dense(512, bc_traj_dim)
)

# ── Dataset Preparation (History + Residual Target) ─────────────────────
function bc_prepare_dataset(dataset, μ_q, σ_q, μ_u, σ_u)
    X_state = Vector{Float32}[]
    Y_traj  = Vector{Float32}[]

    # dataset is a list of (history_vector_148, target_U_480)
    for (history_state, target_u) in dataset
        qn = vec((history_state .- μ_q) ./ σ_q)
        un = vec((target_u .- μ_u) ./ σ_u)

        push!(X_state, Float32.(qn))
        push!(Y_traj,  Float32.(un))
    end

    return hcat(X_state...), hcat(Y_traj...)
end

# ── Training ────────────────────────────────────────────────────────────
function train_bc_policy!(policy, X_state, Y_traj; epochs=120, batchsize=32, lr=5e-4)
    opt = Flux.setup(Adam(lr), policy)
    loader = DataLoader((X_state, Y_traj), batchsize=batchsize, shuffle=true)
    N = size(X_state, 2)

    println("\n=== Training History-Conditioned MLP Policy ($epochs Epochs, $N samples, BS $batchsize) ===")

    best_loss = Inf

    for epoch in 1:epochs
        total_loss = 0.0f0
        num_batches = 0

        for (s_b, u_b) in loader
            loss, grads = Flux.withgradient(policy) do m
                u_pred = m(s_b)
                mean((u_pred .- u_b).^2)
            end

            Flux.update!(opt, policy, grads[1])
            total_loss += loss
            num_batches += 1
        end

        avg_loss = total_loss / num_batches
        best_loss = min(best_loss, avg_loss)

        if epoch % 10 == 0 || epoch == 1 || epoch == epochs
            println("Epoch $(lpad(epoch, 3)) / $epochs | MSE = $(round(avg_loss, digits=6)) | Best = $(round(best_loss, digits=6))")
        end
    end

    println("✓ MLP Policy Training Complete! Best MSE = $(round(best_loss, digits=6))\n")
end

# ── Inference: Single Forward Pass with History ─────────────────────────
function bc_predict(policy, history_state, μ_q, σ_q, μ_u, σ_u)
    norm_state = Float32.((history_state .- μ_q) ./ σ_q)
    norm_traj  = policy(norm_state)
    μ_u_mat = reshape(μ_u, bc_nu, bc_H)
    σ_u_mat = reshape(σ_u, bc_nu, bc_H)
    traj_delta = reshape(norm_traj, bc_nu, bc_H)
    unnorm_delta = (traj_delta .* σ_u_mat) .+ μ_u_mat
    
    # Residual Output: HOME_POS baseline + unnorm_delta
    return HOME_POS .+ unnorm_delta
end
