
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

function plot_loggingwrapper(env; plot_std=true)
    info = get_info(env)["LoggingWrapper"]

    K = 0.002 * 3600 / 1000 # 0.002 m^2, 3600 s/hr, micro to mili
    x, y_mean, y_std = get_mean_std(info["steps"], K*info["reward"])
    if plot_std
        p1 = plot(x, y_mean+y_std; fillrange=y_mean-y_std, label=false, fillalpha = 0.35, linewidth=0)
    else
        p1 = plot()
    end
    p1 = plot!(p1, x, y_mean, label=false, title="Mean Episodic Return [mmol CO2]")

    x, y_mean, y_std = get_mean_std(info["steps"], info["episode_length"])
    if plot_std
        p2 = plot(x, y_mean+y_std; fillrange=y_mean-y_std, label=false, fillalpha = 0.35, linewidth=0)
    else
        p2 = plot()
    end
    p2 = plot!(p2, x, y_mean, label=false, title="Mean Episode Length [hours]", xlabel="Steps")

    plot(p1, p2, layout=(2,1))
end

env = VecEnv(n_envs=8) do 
    PlantEcoPhys_RL.PhotonsynEnv(; Esoil_init_max = 100., n_obs=2, max_steps=24*30)
end |> LoggingWrapper
env = RewNorm(; env)
# env = ObsNorm(; env)

solver = PPOSolver(; env, 
    ac=ContinuousActorCritic(env; squash=true, actor_dims=[128], critic_dims=[256], shared_dims=256), 
    discount=1f0, 
    n_steps=10_000_000, 
    lr_decay=false,
    batch_size=128,
    traj_len=1024,
    n_epochs=4,
    gae_lambda=0.99
)
ac, info = solve(solver)

plot_loggingwrapper(env; plot_std=false)
savefig("mdp_training_undiscounted.png")

function plot_info(info)
    plts = [plot(val[1],vcat(val[2]...); ylabel=key, label=false) for (key,val) in info]
    plot(plts...)
end
plot_info(info)

env = PhotonsynEnv(; Esoil_init_max = 100., n_obs=2)
reset!(env)
env.internal.Esoil = 50.
env.idx = 12
env.internal.GS = 0.1
PlantEcoPhys_RL.PhotosynEB(env.internal, env.external_trajectory[env.idx])
env.internal.Esoil

env = PlantEcoPhys_RL.PhotonsynEnv(; Esoil_init_max = 100., n_obs=2, max_steps=24*30)
reset!(env)
env.idx = 1
env.internal.Esoil = 300.
gs = Float64[]
Esoil = Float64[]
A = Float64[]
Tleaf = Float64[]
while !terminated(env)
    s = observe(env)
    a = ac(s)
    r = act!(env, a)
    push!(gs, env.internal.GS)
    push!(A, r)
    push!(Esoil, env.internal.Esoil)
    push!(Tleaf, env.internal.Tleaf)
end
Esoil = PlantEcoPhys_RL.transpiration2loss.(Esoil, 0.002)/1000
p1 = plot(gs, label=false, title="Conductance [mol / m2 / s]")
p2 = plot(Esoil, label=false, title="Water Remaining [g]")
p3 = plot(A, label=false, title="Rate of Photosynthesis [umol / m2 / s]")
p4 = plot(df.GS[1:length(A)], label=false, title="'True' Conductance [mol / m2 / s]")
VPD = [env.external_trajectory[i].VPD for i in 1:length(A)]
p5 = plot(VPD; title="VPD [kPa]", label=false, xlabel="Time [Hours]")
PPFD = [env.external_trajectory[i].PPFD for i in 1:length(A)]
p6 = plot(PPFD; title="PPFD [umol/m2/s]", label=false, xlabel="Time [Hours]")
Tair = [env.external_trajectory[i].Tair for i in 1:length(A)]
p7 = plot(Tair; title="Tair [C]", label=false, xlabel="Time [Hours]")
p8 = plot(Tair - Tleaf; title="Tair - Tleaf [C]", label=false, xlabel="Time [Hours]")

p11 = plot(
    p4, p1, p3, p2, p5, p6, p7, p8,
    xlabel="Time [Hours]",
    layout=(4,2),
    size = (900,750)
)
savefig("sample_traj.png")

p12 = plot(p1,p3,p2, layout=(3,1), size=(600,500))
savefig("sample_traj.png")

plot(p11, p12, size=(900,600))
savefig("sample_traj.png")

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


