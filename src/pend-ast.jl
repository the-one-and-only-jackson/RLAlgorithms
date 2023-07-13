using Revise

includet("pendulum.jl")

using RLAlgorithms.MultiEnv
using RLAlgorithms.PPO

using CommonRLInterface.Wrappers

struct AST_distributional <: Wrappers.AbstractWrapper
    env::AbstractEnv
    t_max::Float64
    terminal_cost::Float64
    step_vec::Vector
    likelihood_vec::Vector
    kl_vec::Vector
    fail_vec::Vector
end
AST_distributional(; env, t_max, terminal_cost) = AST_distributional(env, t_max, terminal_cost, Int64[], Float64[], Float64[], Bool[])
Wrappers.wrapped_env(e::AST_distributional) = e.env
function CommonRLInterface.reset!(e::AST_distributional)
    if terminated(e.env) && !isempty(e.fail_vec)
        e.fail_vec[end] = true
    end
    if !isempty(e.likelihood_vec)
        steps_remaining = ceil((e.t_max - e.env.t) / e.env.dt)
        max_likelihood = loglikelihood(actions(e.env).d, actions(e.env).d.μ)
        e.likelihood_vec[end] += steps_remaining * max_likelihood
    end

    push!(e.step_vec, 0)
    push!(e.likelihood_vec, 0.0)
    push!(e.kl_vec, 0.0)
    push!(e.fail_vec, false)

    reset!(e.env)
    nothing
end
function CommonRLInterface.act!(e::AST_distributional, a)
    model_distribution = actions(e.env).d

    N = length(a) ÷ 2
    if N == 1
        μ = model_distribution.μ + model_distribution.σ * a[1]
        σ = model_distribution.σ * exp(a[2] / 2)
        d = Normal(μ, σ)
    else
        μ = model_distribution.μ + model_distribution.Σ * a[1:N]
        Σ = model_distribution.Σ * exp(a[N+1:end] / 2) # ?
        d = MvNormal(μ, Σ)
    end

    x = rand(d)
    act!(e.env, x)

    kl = kl_divergence(d, model_distribution)
    likelihood = loglikelihood(model_distribution, x)

    timeout = terminated(e) && !terminated(e.env)
    r = likelihood - e.terminal_cost * timeout

    # logging
    e.step_vec[end] += 1
    e.likelihood_vec[end] += likelihood
    e.kl_vec[end] += kl

    return r
end
function kl_divergence(p::Normal, q::Normal)
    return log(q.σ/p.σ) + (p.σ^2 + (p.μ-q.μ)^2)/(2*q.σ^2) - 1//2
end
function CommonRLInterface.actions(e::AST_distributional)
    N = 2 * length(actions(e.env))
    Box(fill(-Inf32,N), fill(Inf32,N))
end
CommonRLInterface.terminated(e::AST_distributional) = terminated(e.env) || e.env.t >= e.t_max


using Plots

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

p1, p2, p3 = plot(), plot(), plot()
for seed in 0:4
    env = MultiEnv.VecEnv(n_envs=8) do 
        env = PendSim()
        AST_distributional(; env, t_max=1.0, terminal_cost=10_000)
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
        seed = seed,
        kl_targ = 0.02
    )

    ac, info = solve(solver)

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

