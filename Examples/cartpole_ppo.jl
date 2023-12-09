using RLAlgorithms.Algorithms: solve, PPOSolver
using RLAlgorithms.MultiEnv: VecEnv, LoggingWrapper, RewNorm, ObsNorm
using RLAlgorithms.Environments: CartPole
using RLAlgorithms.CommonRLExtensions: get_info

# Input is a function to output a CommonRLInterface.AbstractEnv
# Equiv to VecEnv(()->Pendulum(); n_envs=8)
discount = 0.99

p = plot(xlabel="Steps", title="Episodic Reward")
for i in 1:5
    solver = PPOSolver(; 
        env = ObsNorm(; env = LoggingWrapper(; discount, env = VecEnv(()->CartPole(); n_envs=8))), 
        discount, 
        n_steps=100_000,
        traj_len = 128,
        batch_size = 128,
        n_epochs = 4,
        kl_targ = 0.02,
        ent_coef = 0.0f0,
        vf_coef = 1.,
        lr=3e-4,
        clip_coef = 0.2,
        ac_kwargs = (; critic_type = :categorical, categorical_values = 0:ceil(Int, 1/(1-discount)))
    )

    ac, info_log = solve(solver)

    hist = get_info(solver.env)["LoggingWrapper"]
    CIplot!(p, hist["steps"], hist["reward"], label=false, c=i)
end
plot(p)


solver = PPOSolver(; 
    env = ObsNorm(; env = LoggingWrapper(; discount, env = VecEnv(()->CartPole(); n_envs=8))), 
    discount, 
    n_steps=100_000,
    traj_len = 128,
    batch_size = 128,
    n_epochs = 10,
    kl_targ = 0.02,
    ent_coef = 0.0f0,
    vf_coef = 1.,
    lr=3e-4,
    clip_coef = 0.2,
    gae_lambda = 1.0,
    ac_kwargs = (; critic_dims = [256, 256], critic_type = :categorical, categorical_values = 0:1+ceil(Int, 1/(1-discount)))
)

ac, info_log = solve(solver)


hist = get_info(solver.env)["LoggingWrapper"]
CIplot(hist["steps"], hist["reward"], label=false, c=1)


plot(info_log[:value_loss]...; label=false, yaxis=:log)

plot(info_log[:kl_est]...; label=false)
info_log

# ==================================================
# Plots
# ==================================================

using Plots, Statistics

# Plotting confidence intervals
CIplot(xdata, ydata; args...) = CIplot!(plot(), xdata, ydata; args...)
function CIplot!(p, xdata, ydata; Nx=500, z=1.96, k=5, c=1, label=false, plotargs...)
    dx = (maximum(xdata)-minimum(xdata))/Nx
    x = (minimum(xdata) + dx/2) .+ dx*(0:Nx-1)
    y = zeros(Nx)
    dy = zeros(Nx)
    for i in eachindex(x)
        y_tmp = ydata[(x[i]-dx*(1/2+k)) .≤ xdata .≤ (x[i]+dx*(1/2+k))]
        y[i] = mean(y_tmp)
        dy[i] = z*std(y_tmp)/sqrt(length(y_tmp))
    end
    plot!(p, x, y-dy; fillrange=y+dy, fillalpha=0.3, c, alpha=0, label=false)
    plot!(p, x, y; c, label, plotargs...)
    return p
end

# Debugging / Loss Curves
plot(
    [plot(val[1], val[2]; xlabel="Steps", title=key, label=false) for (key,val) in info_log]...; 
    size=(900,900)
)

# Reward / Learning Curves
hist = get_info(solver.env)["LoggingWrapper"]
p = plot(xlabel="Steps", title="Episodic Reward")
CIplot!(p, hist["steps"], hist["reward"], label="Undiscounted Reward", c=1)
CIplot!(p, hist["steps"], hist["discounted_reward"], label="Discounted Reward", c=2)


# ==================================================
# Test a sample trajectory from the learned actor
# ==================================================

using CommonRLInterface

μ = get_info(solver.env)["ObsNorm"]["mean"]
σ = get_info(solver.env)["ObsNorm"]["std"]

test_env = Pendulum()

reset!(test_env)
s_vec, a_vec = [], []
for _ in 1:100
    s = observe(test_env)
    a = ac((s - μ) ./ σ)
    act!(test_env, a)
    push!.((s_vec, a_vec), (s, a))
end

theta = atan.(stack(s_vec)[2,:], stack(s_vec)[1,:])
theta_dot = stack(s_vec)[3,:]
a = clamp.(stack(a_vec)[:], -2, 2)

plot(
    plot(theta; ylabel="θ", label=false, linetype=:steppre),
    plot(theta_dot; ylabel="dθdt", label=false, linetype=:steppre),
    plot(a; ylabel="a", xlabel="Steps", label=false, linetype=:steppre);
    layout=(3,1)
)
