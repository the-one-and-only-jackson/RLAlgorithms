using RLAlgorithms.Environments
using RLAlgorithms.MultiEnv
using RLAlgorithms.Algorithms
using RLAlgorithms.CommonRLExtensions
using CommonRLInterface
using Plots
using Statistics

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

function plot_loggingwrapper(env; plot_std=true)
    info = get_info(env)["LoggingWrapper"]

    x, y_mean, y_std = get_mean_std(info["steps"], info["reward"])
    if plot_std
        p1 = plot(x, y_mean+y_std; fillrange=y_mean-y_std, label=false, fillalpha = 0.35, linewidth=0)
    else
        p1 = plot()
    end
    p1 = plot!(p1, x, y_mean, label=false, title="Mean Episodic Return")

    x, y_mean, y_std = get_mean_std(info["steps"], info["discounted_reward"])
    if plot_std
        p2 = plot(x, y_mean+y_std; fillrange=y_mean-y_std, label=false, fillalpha = 0.35, linewidth=0)
    else
        p2 = plot()
    end
    p2 = plot!(p2, x, y_mean, label=false, title="Mean Discounted Episodic Return")

    x, y_mean, y_std = get_mean_std(info["steps"], info["episode_length"])
    if plot_std
        p3 = plot(x, y_mean+y_std; fillrange=y_mean-y_std, label=false, fillalpha = 0.35, linewidth=0)
    else
        p3 = plot()
    end
    p3 = plot!(p3, x, y_mean, label=false, title="Mean Episode Length", xlabel="Steps")

    plot(p1, p2, p3, layout=(3,1))
end

discount = 0.9f0

env = VecEnv(n_envs=1) do 
    Pendulum()
end
env = RewNorm(; env=LoggingWrapper(; env, discount=discount), gamma=discount)

solver = PPOSolver(; env,
    traj_len = 2048,
    batch_size = 64,
    n_epochs = 10,
    n_steps = 10_000_000,
    discount,
    gae_lambda = 0.1,
    lr = 3f-4,
    lr_decay=false,
    ac=ContinuousActorCritic(env; squash=true)
)

ac, info_log = solve(solver)

plot_loggingwrapper(env)

function plot_info(info)
    plts = [plot(val[1],vcat(val[2]...); ylabel=key, label=false) for (key,val) in info]
    plot(plts...)
end
plot_info(info_log)

ac

test_env = Pendulum()
reset!(test_env)
while !truncated(env) && !terminated(env)
    s = observe(test_env) |> ac


solver.ac.log_std

