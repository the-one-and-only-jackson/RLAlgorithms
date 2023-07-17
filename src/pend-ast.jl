using Revise

using RLAlgorithms.MultiEnv: VecEnv
using RLAlgorithms.PPO: solve

includet("pendulum.jl")
using Pendulum: PendSim

includet("ast.jl")
using AST: AST_distributional

using Plots: plot, plot!, savefig

function get_mean_std(x_data, y_data; nx = 500, k = 5)
    x = (1:nx) * ceil(maximum(x_data)/nx)
    y_mean = zeros(size(x))
    y_std = zeros(size(x))
    
    for ii = 1:nx
        xmin = (ii-k)<1 ? 0 : x[ii-k]
        xmax = (ii+k)>nx ? x[end] : x[ii+k]
        idxs = xmin .< x_data .<= xmax
        y_mean[ii] = y_data[idxs] |> mean
        y_std[ii] = y_data[idxs] |> std
    end

    return x, y_mean, y_std
end

function run_exp()
    env = VecEnv(n_envs=8) do 
        AST_distributional(; env=PendSim(), n_steps=100)
    end

    solver = PPOSolver(;
        env,
        n_steps = 2_500_000,
        lr = 3f-4,
        lr_decay = false,
        traj_len = 128,
        batch_size = 128,
        n_epochs = 10,
        discount = 0.99f0,
        gae_lambda = 1f0, # 1.0 corresponds to not using GAE
        norm_advantages = true,
        seed = 0,
        kl_targ = 0.02
    )

    ac, info = solve(solver)

    return env, ac, info
end


## this still needs to be edited
p1, p2, p3 = plot(), plot(), plot()
for seed in 0:4


    step_vec = Int64[]
    likelihood_vec = Float64[]
    kl_vec = Float64[]
    fail_vec = Bool[]
    for e in envs(env)
        append!(step_vec, length(env)*cumsum(e.step_vec))
        append!(likelihood_vec, e.likelihood_vec)
        append!(kl_vec, e.kl_vec)
        append!(fail_vec, e.fail_vec)
    end

    x, y_mean, y_std = get_mean_std(step_vec, likelihood_vec; k=10)
    plot!(p1, x, y_mean, label=false, xlabel="Steps", title="Sum Log Likelihood")

    x, y_mean, y_std = get_mean_std(step_vec, kl_vec; k=10)
    plot!(p2, x, y_mean, label=false, xlabel="Steps", title="Sum KL-Divergence")

    x, y_mean, y_std = get_mean_std(step_vec, fail_vec; k=10)
    plot!(p3, x, y_mean, label=false, xlabel="Steps", title="Fail Rate")
end
plot(p1)
savefig("src/fig/likelihood-discount.png")
plot(p2)
savefig("src/fig/kl-discount.png")
plot(p3)
savefig("src/fig/fail-discount.png")


plot_vec = []
for (key, val) in info
    push!(plot_vec, plot(val[1], val[2], title=string(key), label=false))
end
plot!(plot_vec...)

ac

