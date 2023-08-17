# julia --project=@. pend-ast.jl

using Revise

using RLAlgorithms.MultiEnv: VecEnv
using RLAlgorithms.PPO: solve, PPOSolver
using RLAlgorithms.MultiEnvWrappers: ObsNorm, RewNorm

include("pendulum.jl")
using .Pendulums: PendSim

include("ast.jl")
using .AST: AST_distributional

using BSON: @save
using Dates: today

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

using CommonRLInterface
RL = CommonRLInterface
"""
FrameSkipWrapper
"""
struct FrameSkipWrapper <: RL.Wrappers.AbstractWrapper
    env::AbstractEnv
    n_frames::Int
    discount::AbstractFloat
end
RL.Wrappers.wrapped_env(e::FrameSkipWrapper) = e.env

function CommonRLInterface.act!(e::FrameSkipWrapper, a)
    r = act!(e.env, a)
    for i = 1:e.n_frames-1
        terminated(env) && break
        r += e.discount^i * act!(e.env, a)
    end
    return r
end

exp_dir = get_exp_dir()

for _ in 0:4
    env = VecEnv(n_envs=8) do 
        e = AST_distributional(; env=PendSim(), n_steps=100, terminal_cost=1000)
        FrameSkipWrapper(e, 1, 1f0)
    end |> ObsNorm

    env = RewNorm(; env)

    solver = PPOSolver(;
        env,
        n_steps = 1_000_000,
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
    @save joinpath(dir, "env.bson") env
    @save joinpath(dir, "ac.bson") ac
    @save joinpath(dir, "info.bson") info
end

env = VecEnv(n_envs=8) do 
    e = AST_distributional(; env=PendSim(), n_steps=100, terminal_cost=1000)
    FrameSkipWrapper(e, 1, 1f0)
end |> ObsNorm

CommonRLInterface.observe(env::PendSim) = Float32.(env.p.x)

env = PendSim()

reset!(env)
observe(env)

env = RewNorm(; env)

using RLAlgorithms.MultiEnv: single_observations
single_observations(env)


env = VecEnv(n_envs=8) do 
    e = AST_distributional(; env=PendSim(), n_steps=100, terminal_cost=1000)
    # FrameSkipWrapper(e, 1, 1f0)
end # |> ObsNorm

# env = RewNorm(; env)

solver = PPOSolver(;
    env,
    n_steps = 1_000_000,
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

using RLAlgorithms.PPO
reset!(env)
buffer = PPO.Buffer(env, solver.traj_len)
@time PPO.rollout!(env, buffer, solver.ac, 1, solver.device)


struct MyStruct
    # stuff...
    nothing_or_arr::Union{Nothing, AbstractArray{Bool}}
end

# VS

abstract type MyType end

struct MyStruct1 <: MyType
    # stuff...
end

struct MyStruct2 <: MyType
    # stuff...
    nothing_or_arr::AbstractArray{Bool}
end
