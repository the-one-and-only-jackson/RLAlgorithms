using Revise

using Statistics: mean, std
using Plots: plot, plot!, savefig

using RLAlgorithms.MultiEnv: VecEnv
using RLAlgorithms.PPO: solve, PPOSolver
using RLAlgorithms.MultiEnvWrappers: ObsNorm, RewNorm, unwrapped

includet("pendulum.jl")
using .Pendulums: PendSim

includet("ast.jl")
using .AST: AST_distributional


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

function run_exp(; seed=0, terminal_cost=1000)
    env = VecEnv(n_envs=8) do 
        AST_distributional(; env=PendSim(), n_steps=100, terminal_cost)
    end |> ObsNorm

    env = RewNorm(; env, gamma=0.99)

    solver = PPOSolver(;
        env,
        n_steps = 1_000_000,
        lr = 3f-4,
        lr_decay = false,
        vf_coef = 1,
        traj_len = 128,
        batch_size = 128,
        n_epochs = 10,
        discount = 0.99f0,
        gae_lambda = 0.95f0, # 1.0 corresponds to not using GAE
        norm_advantages = true,
        seed,
        kl_targ = 0.02
    )

    ac, info = solve(solver)

    return env, ac, info
end

# plot([plot(x,y; label=false, title) for (title,(x,y)) in info]...)

for terminal_cost in [500, 1000, 5000, 10000]
    results = [run_exp(; seed, terminal_cost) for seed in 0:9]

    p = plot()
    for (env,_,_) in results
        env_info = Dict()
        for e in unwrapped(env).envs, (key, val) in e.info
            dst = get!(env_info, key, eltype(val)[])
            if key == :steps
                val = length(env)*cumsum(val)
            end
            append!(dst, val)
        end

        x, y_mean, y_std = get_mean_std(env_info[:steps], env_info[:fail]; k=10)
        plot!(p, x, y_mean, label=false, xlabel="Steps", title="Fails")
    end
    savefig(p, "src/fig/fail_cost_$terminal_cost.png")
end

function test_fail(env, n_steps)
    reset!(env)
    for _ in 1:n_steps
        a = actions(env).d |> rand
        act!(env, a)
        terminated(env) && return true
    end
    return false
end
mean(test_fail(PendSim(), 100) for _ in 1:10_000)

v = [sum(0.99^j for j in 0:i) for i = 0:99]
mean(v)
std(v)
# do reward scaling
r_mean = -entropy(dist)
(r - r_mean*mean(v)) / (mean(v)*std(v))

####

# Notes:
# GAE is huge! 0.95 performs much better than 1.0


results = [run_exp(; seed, terminal_cost=1_000) for seed in 0:9]

for key in [:fail, :likelihood, :KL]
    p = plot()
    for (env,_,_) in results
        env_info = Dict()
        for e in unwrapped(env).envs, (key, val) in e.info
            dst = get!(env_info, key, eltype(val)[])
            if key == :steps
                val = length(env)*cumsum(val)
            end
            append!(dst, val)
        end

        x, y_mean, y_std = get_mean_std(env_info[:steps], env_info[key]; k=10)
        plot!(p, x, y_mean, label=false, xlabel="Steps", title=key)
    end
    display(p)
end


using CommonRLInterface
using Distributions
-entropy(actions(PendSim()).d) * 100

x0 = rand(MvNormal([1, 0.1, 1, 1]))

env = AST_distributional(; env=PendSim(; x0=Dirac(x0)), n_steps=100, terminal_cost=1000)
ac = results[5][2]
reset!(env)
s_vec, a_vec, r_vec = [], [], []
while !terminated(env)
    s = observe(env) .|> Float32
    a = ac(s)
    r = act!(env, a)
    push!(s_vec, s)
    push!(a_vec, a)
    push!(r_vec, r)
    nothing
end
a_dist_vec = [AST.unflatten(actions(env.env).d, a) for a in a_vec]
μ = mean.(a_dist_vec)
σ = std.(a_dist_vec)
plot(μ+σ; fillrange=μ-σ)


####

x, y_mean, y_std = get_mean_std(step_vec, likelihood_vec; k=10)
plot!(p1, x, y_mean, label=false, xlabel="Steps", title="Sum Log Likelihood")



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

