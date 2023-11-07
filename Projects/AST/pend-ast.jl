# julia --project=@. pend-ast.jl

using RLAlgorithms.MultiEnv: VecEnv, single_observations, single_actions
using RLAlgorithms.PPO
using RLAlgorithms.MultiEnvWrappers

using CommonRLInterface
const RL = CommonRLInterface

include("pendulum.jl")
using .Pendulums: PendSim

include("ast.jl")
using .AST: AST_distributional

using BSON
using Dates: today
using Plots
using Statistics


function get_exp_dir()
    ii=1
    while isdir(joinpath("Experiments", "$(today())_$ii"))
        ii += 1
    end
    dir = joinpath("Experiments", "$(today())_$ii")
    mkpath(dir)
    cp(@__FILE__, joinpath(dir, "experiment.jl"))
    return dir
end

function get_sub_dir(main_dir)
    ii=1
    while isdir(joinpath(main_dir, "$ii"))
        ii += 1
    end
    return mkpath(joinpath(main_dir, "$ii"))
end

"""
FrameSkipWrapper
"""
struct FrameSkipWrapper{E<:AbstractEnv} <: RL.Wrappers.AbstractWrapper
    env::E
    n_frames::Int
    discount::Float64
end
RL.Wrappers.wrapped_env(e::FrameSkipWrapper) = e.env

function CommonRLInterface.act!(e::FrameSkipWrapper, a)
    r = 0.0
    for i = 0:e.n_frames-1
        r += e.discount^i * act!(e.env, a)
        terminated(e.env) && break
    end
    return r
end

for frame_skip in [1,2,4,10]
    exp_dir = get_exp_dir()

    for _ in 0:4
        env = VecEnv(n_envs=8) do 
            e = AST_distributional(; env=PendSim(), n_steps=100, terminal_cost=1000)
            FrameSkipWrapper(e, frame_skip, 1.0)
        end
        env = ObsNorm(; env)
        env = RewNorm(; env)

        solver = PPOSolver(;
            env,
            n_steps = Int(10_000_000 / frame_skip),
            lr = 3f-4,
            lr_decay = true,
            vf_coef = 1,
            traj_len = 128,
            batch_size = 128,
            n_epochs = 4,
            discount = 1,
            gae_lambda = 0.95f0, # 1.0 corresponds to not using GAE
            norm_advantages = true,
            kl_targ = 0.02
        )

        ac, info = solve(solver)

        dir = get_sub_dir(exp_dir)
        BSON.@save joinpath(dir, "env.bson") env
        BSON.@save joinpath(dir, "ac.bson") ac
        BSON.@save joinpath(dir, "info.bson") info
    end
end



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

function get_data_all_seed(env_vec)
    env_info = Dict()
    for wrapped_vec_env in env_vec
        vec_env = unwrapped(wrapped_vec_env)
        L = length(vec_env)
        for e in vec_env.envs, (key, val) in AST.info(e.env) # this should be unwrapped too
            dst = get!(env_info, key, eltype(val)[])
            if key == :steps
                val = L*cumsum(val)
            end
            append!(dst, val)
        end
    end
    I = sortperm(env_info[:steps])
    for (key,val) in env_info
        env_info[key] = val[I]
    end
    return env_info
end

function plot_mean_std(x_data, y_data, p, x_interp_tup=nothing; plot_mean=true, k=20, label=false, plot_kw...)
    x, y_mean, y_std = get_mean_std(x_data, y_data; k)

    if !isnothing(x_interp_tup)
        new_x = zeros(size(x))
        for ii = 1:length(x)
            jj = findfirst(x_interp_tup[1] .> x[ii])
            if isnothing(jj)
                new_x[ii:end] .= x_interp_tup[2][end]
                break
            end
            new_x[ii] = x_interp_tup[2][jj]
        end
        x = new_x
    end
    
    if plot_mean
        plot!(p, x, y_mean+y_std; fillrange=y_mean-y_std, label=false, fillalpha = 0.35, linewidth=0, plot_kw...)
    end
    plot!(p, x, y_mean; label, plot_kw...)
end

function get_mean_std(x_data, y_data; nx = 500, k = 5)
    x = (1:nx) * ceil(maximum(x_data)/nx)
    y_mean, y_std = zeros(size(x)), zeros(size(x))
    for ii = 1:nx
        xmin = (ii-k)<1 ? 0 : x[ii-k]
        xmax = (ii+k)>nx ? x[end] : x[ii+k]
        idxs = xmin .< x_data .<= xmax
        y_mean[ii] = y_data[idxs] |> mean
        y_std[ii] = y_data[idxs] |> std
    end
    return x, y_mean, y_std
end


exp_files = ["2023-08-25_5", "2023-08-25_6", "2023-08-25_7", "2023-08-25_8"]


fail_plot = plot(xlabel="Steps", ylabel="Fail Rate")
likelihood_plot = plot(xlabel="Steps", ylabel="Likelihood")
kl_plot = plot(xlabel="Steps", ylabel="Total KL")
labels = [1, 2, 4, 10]
for (ii,exp_file) in enumerate(exp_files)
    exp_dir = joinpath("Experiments", exp_file)
    exp_data = load_exp(exp_dir)
    env_info = get_data_all_seed(exp_data[:env])
    k = 10
    plot_mean = false
    plot_mean_std(env_info[:steps], env_info[:fail], fail_plot; plot_mean, k, c=ii, label=labels[ii])
    plot_mean_std(env_info[:steps], env_info[:likelihood], likelihood_plot; plot_mean, k, c=ii, label=labels[ii])
    plot_mean_std(env_info[:steps], env_info[:KL], kl_plot; plot_mean, k, c=ii, label=labels[ii])
end
plot(fail_plot, likelihood_plot, kl_plot, size=(700,600))
savefig("frameskip_step.png")

fail_plot = plot(xlabel="Time (s)", ylabel="Fail Rate")
likelihood_plot = plot(xlabel="Time (s)", ylabel="Likelihood")
kl_plot = plot(xlabel="Time (s)", ylabel="Total KL")
labels = [1, 2, 4, 10]
for (ii,exp_file) in enumerate(exp_files)
    exp_dir = joinpath("Experiments", exp_file)
    exp_data = load_exp(exp_dir)
    env_info = get_data_all_seed(exp_data[:env])
    k = 10
    plot_mean = false
    x_tup = exp_data[:info][1][:wall_time]
    x_tup[1] .*= labels[ii]
    plot_mean_std(env_info[:steps], env_info[:fail], fail_plot, x_tup; plot_mean, k, c=ii, label=labels[ii])
    plot_mean_std(env_info[:steps], env_info[:likelihood], likelihood_plot, x_tup; plot_mean, k, c=ii, label=labels[ii])
    plot_mean_std(env_info[:steps], env_info[:KL], kl_plot, x_tup; plot_mean, k, c=ii, label=labels[ii])
end
plot(fail_plot, likelihood_plot, kl_plot, size=(700,600))
savefig("frameskip_time.png")





