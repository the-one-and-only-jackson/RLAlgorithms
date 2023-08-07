# julia --project=@. cwh-ast.jl

using RLAlgorithms.MultiEnv: VecEnv
using RLAlgorithms.PPO: solve, PPOSolver
using RLAlgorithms.MultiEnvWrappers: ObsNorm, RewNorm

include("cwh.jl")
using .Satellites: CWHSim

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

exp_dir = get_exp_dir()

for seed in 0:4
    env = VecEnv(n_envs=8) do 
        AST_distributional(; env=CWHSim(), n_steps=500, terminal_cost=5000)
    end |> ObsNorm

    env = RewNorm(; env)

    solver = PPOSolver(;
        env,
        n_steps = 5_000_000,
        lr = 3f-4,
        lr_decay = true,
        vf_coef = 1,
        traj_len = 128,
        batch_size = 128,
        n_epochs = 10,
        discount = 0.99,
        gae_lambda = 0.95f0, # 1.0 corresponds to not using GAE
        norm_advantages = true,
        seed,
        kl_targ = 0.02
    )

    ac, info = solve(solver)

    dir = get_sub_dir(exp_dir)
    @save joinpath(dir, "env.bson") env
    @save joinpath(dir, "ac.bson") ac
    @save joinpath(dir, "info.bson") info
end



# using CommonRLInterface
# env = AST_distributional(; env=CWHSim(), n_steps=500, terminal_cost=5000)

# function test_env(env)
#     x = randn(Float32, 2)
#     reset!(env)
#     for _ in 1:500
#         a = [x; zeros(Float32, 2)]
#         terminated(env) && return true
#         act!(env, a)
#     end
#     return false
# end

# count(test_env(env) for _ in 1:500)/500


env = VecEnv(n_envs=8) do 
    AST_distributional(; env=CWHSim(), n_steps=500, terminal_cost=5000)
end |> ObsNorm

env = RewNorm(; env)

solver = PPOSolver(;
    env,
    n_steps = 5_000_000,
    lr = 3f-4,
    lr_decay = true,
    vf_coef = 1,
    traj_len = 128,
    batch_size = 128,
    n_epochs = 10,
    discount = 0.99,
    gae_lambda = 0.95f0, # 1.0 corresponds to not using GAE
    norm_advantages = true,
    seed = 0,
    kl_targ = 0.02
)

ac, info = solve(solver)