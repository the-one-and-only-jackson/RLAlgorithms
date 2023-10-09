
include("PlantEcoPhys_RL.jl")
using .PlantEcoPhys_RL
using CommonRLInterface
using RLAlgorithms.MultiEnv
using RLAlgorithms.Algorithms
using RLAlgorithms.CommonRLExtensions
using Plots
using Statistics

# 0.002 m2 = 20 cm2

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

function plot_loggingwrapper(env)
    info = get_info(env)["LoggingWrapper"]

    x, y_mean, y_std = get_mean_std(info["steps"], info["reward"])
    p1 = plot(x, y_mean+y_std; fillrange=y_mean-y_std, label=false, fillalpha = 0.35, linewidth=0)
    p1 = plot!(p1, x, y_mean, label=false, ylabel="Reward")

    x, y_mean, y_std = get_mean_std(info["steps"], info["episode_length"])
    p2 = plot(x, y_mean+y_std; fillrange=y_mean-y_std, label=false, fillalpha = 0.35, linewidth=0)
    p2 = plot!(p2, x, y_mean, label=false, ylabel="Episode Length", xlabel="Steps")

    plot(p1, p2, layout=(2,1))
end

env = VecEnv(n_envs=8) do 
    PhotonsynEnv(; Esoil_init_max = 100., n_obs=2)
end |> LoggingWrapper
env = RewNorm(; env)
# env = ObsNorm(; env)

solver = PPOSolver(; env, 
    ac=ContinuousActorCritic(env; squash=true), 
    discount=0.99f0, 
    n_steps=10_000_000, 
    lr_decay=true,
    batch_size=128,
    traj_len=1024,
    n_epochs=4,
    gae_lambda=0.95
)
ac, info = solve(solver)

plot_loggingwrapper(env)
savefig("init_training_discounted.png")

function plot_info(info)
    plts = [plot(val[1],vcat(val[2]...); ylabel=key, label=false) for (key,val) in info]
    plot(plts...)
end
plot_info(info)


env = PhotonsynEnv(; Esoil_init_max = 100., n_obs=2)
reset!(env)
env.idx = 1
env.internal.Esoil = 50.
gs = Float64[]
Esoil = Float64[]
while !terminated(env)
    s = observe(env)
    a = ac(s)
    act!(env, a)
    push!(gs, env.internal.GS)
    push!(Esoil, env.internal.Esoil)
end
plot(
    plot(gs, label=false, ylabel="Rate of Photosynthesis"), 
    plot(Esoil, label=false, ylabel="Water Remaining (g)"),
    xlabel="Hours",
    layout=(1,2)
)
savefig("discounted_traj.png")

push!(gs, 0)
plot(reshape(gs, 24, 10), label=false)

function rand_baseline()
    env = PhotonsynEnv(; Esoil_init_max = 100., n_obs=2)
    reset!(env)
    step = 0
    while !terminated(env)
        a = 2*rand()-1
        act!(env,a)
        step += 1
    end
    return step
end

y = [rand_baseline() for _ in 1:1000]
mean(y)
std(y)




ext_traj = PlantEcoPhys_RL.get_externaltraj()

mean(stack(PlantEcoPhys_RL.struct2vec.(ext_traj)); dims=2)
std(stack(PlantEcoPhys_RL.struct2vec.(ext_traj)); dims=2)

L = length(PlantEcoPhys_RL.struct2vec(ext_traj[1]))
M = zeros(L, length(ext_traj))
for (i,x) in enumerate(ext_traj)
    M[:,i] = PlantEcoPhys_RL.struct2vec(x)
end
mean(M; dims=2)
std(M; dims=2)


ext_traj = PlantEcoPhys_RL.get_externaltraj()
vpd = [x.VPD for x in ext_traj]
mean(vpd)
std(vpd)


reset!(env)
observe(env) |> solver.ac
(ans .+ 1)/2

ac
observe(env) |> ac .|> ac2gs
ac2gs(a) = 0.0 + (0.3 - 0.0) * (1 + clamp(a, -one(a), one(a)))/2

CommonRLInterface.Wrappers.unwrapped(env).envs



savefig("PPO_curves.png")

ac.log_std


### 


function baseline_rand()
    env = PhotonsynEnv(; Esoil_init_max = 100.0)
    reset!(env)
    steps = 0
    r = 0.0
    while !terminated(env)
        a = 2*rand()-1
        r += 0.99^steps * act!(env, a)
        steps += 1
    end

    r, steps
end

r_vec = Float64[]
step_vec = Int[]
for _ in 1:1000
    r,step = baseline_rand()
    push!.((r_vec,step_vec), (r,step))
end
r_vec
step_vec


