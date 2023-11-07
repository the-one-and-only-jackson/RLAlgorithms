using Statistics: mean, std
using Plots: plot, plot!, savefig

using RLAlgorithms.MultiEnv: VecEnv, AbstractMultiEnv
using RLAlgorithms.PPO: solve, PPOSolver
using RLAlgorithms.MultiEnvWrappers: ObsNorm, RewNorm, unwrapped

# include("pendulum.jl")
# using .Pendulums: PendSim

include("cwh.jl")
using .Satellites: CWHSim

include("ast.jl")
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

function plot_ppo_info(info)
    plot([plot(x,y; label=false, title) for (title,(x,y)) in info]...)
end

function plot_ast_info(env::AbstractMultiEnv, p_vec=[plot() for _ in 1:3])
    env = unwrapped(env)
    for (ii,key) in enumerate([:fail, :likelihood, :KL])
        env_info = Dict()
        for e in env.envs, (key, val) in e.info
            dst = get!(env_info, key, eltype(val)[])
            if key == :steps
                val = length(env)*cumsum(val)
            end
            append!(dst, val)
        end

        x, y_mean, y_std = get_mean_std(env_info[:steps], env_info[key]; k=10)
        plot!(p_vec[ii], x, y_mean, label=false, xlabel="Steps", title=key)
    end
    return p_vec
end


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

using BSON

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


function evaluate(env, ac; max_steps=1_000)
    info = Dict(:s=>[], :a=>[], :r=>[])
    reset!(env)
    for _ in 1:max_steps
        terminated(env) && break
        s = observe(env) 
        a = ac(s .|> Float32)
        r = act!(env, a)
        for (key,val) in zip([:s,:a,:r],[s,a,r])
            push!(info[key], val)
        end
    end
    return info
end

using CommonRLInterface
using CommonRLInterface.Wrappers: QuickWrapper
using Distributions

x0 = rand(MvNormal([1, 0.1, 1, 1]))

p = plot()
for res in results
    μ = mean(res[1].env.obs_stats) .|> Float32
    σ = sqrt.(var(res[1].env.obs_stats)) .|> Float32
    env = QuickWrapper(
        AST_distributional(; env=PendSim(; x0=Dirac(x0)), n_steps=100, terminal_cost=1000), 
        observe = env -> (observe(env)-μ)./σ
    )
    ac = res[2]
    info = evaluate(env, ac)

    a_dist_vec = [AST.unflatten(actions(env.env.env).d, a) for a in info[:a]]
    μ, σ = mean.(a_dist_vec), std.(a_dist_vec)
    plot!(p, μ+σ; fillrange=μ-σ, fillalpha = 0.35, linewidth=0)
end
display(p)

env.env.info
fail, steps, likelihood, kl

dist = PendSim().noise
v = [sum(loglikelihood(dist, rand(dist)) for _ in 1:100) for _ in 1:1000]
mean(v)
std(v)



plot([plot(x,y; label=false, title) for (title,(x,y)) in results[1][3]]...)
plot(results[1][3][:value_loss], yaxis=:log)

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

### 
using BSON

function load_exp(exp_dir)
    idxs = readdir(exp_dir; join=true) .|> isdir
    N = count(idxs)
    exp_data = Dict()
    for dir in readdir(exp_dir)[idxs]
        path = joinpath(exp_dir, dir)
        for file in readdir(path)
            if length(file)<=5 || file[end-4:end] != ".bson"
                continue
            end
            for (key,val) in BSON.load(joinpath(path, file))
                v = get!(exp_data, key, Vector{Any}(undef, N))
                v[parse(Int,dir)] = val
            end
        end
    end
    return exp_data
end

exp_dir = joinpath("AST", "Experiments", "2023-07-27_1")
exp_data = load_exp(exp_dir)

p_vec = [plot() for _ in 1:3]
for env in exp_data[:env]
    p_vec = plot_ast_info(env, p_vec)
end
for (ii,p) in enumerate(p_vec)
    savefig(p, joinpath(exp_dir, "fig_$ii.png"))
end








using CommonRLInterface
using CommonRLInterface.Wrappers: QuickWrapper
using Distributions

ac = exp_data[:ac][1]
obs_stats = exp_data[:env][1].env.obs_stats
μ = mean(obs_stats) .|> Float32
σ = sqrt.(var(obs_stats)) .|> Float32
env = QuickWrapper(
    AST_distributional(; env=CWHSim(), n_steps=500, terminal_cost=5000), 
    observe = env -> (observe(env)-μ)./σ
)
info = evaluate(env, ac)

a_dist_vec = [AST.unflatten(actions(env.env.env).d, a) for a in info[:a]]

p_vec = []
for (ii, label) in enumerate(["Measurement", "Process"])
    a = [x[ii] for x in mean.(a_dist_vec)]
    b = [x[ii] for x in [sqrt.(x) for x in var.(a_dist_vec)]]
    p = plot(a+b; fillrange=a-b, fillalpha = 0.35, linewidth=0, label)
    push!(p_vec, p)
end
plot(p_vec...; layout=(2,1), xlabel="Time Steps")

s_flat = (σ .* info[:s][5]) + μ



env, ac = exp_data[:env][1], exp_data[:ac][1]
reset!(env)
while !terminated(env)
    s = observe(env)
    a = ac(s .|> Float32)
end

exp_data[:env][1].env.obs_stats