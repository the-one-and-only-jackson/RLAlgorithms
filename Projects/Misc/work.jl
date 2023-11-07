using RLAlgorithms.Spaces
using RLAlgorithms.CommonRLExtensions
using RLAlgorithms.MultiEnv
using RLAlgorithms.Algorithms
using RLAlgorithms.Environments

using CommonRLInterface
using CommonRLInterface.Wrappers
using Plots

using StaticArrays

weights(::MountainCar) = SA[-0.1, 100]

function CommonRLInterface.act!(mc::MountainCar, action::Real; scalar=false)
    mc.steps[] += 1

    force = clamp(action, mc.min_action, mc.max_action)
    dv = force * mc.power - 0.0025 * cos(3 * mc.state[1])
    mc.state[2] = clamp(mc.state[2] + dv, -mc.max_speed, mc.max_speed)
    mc.state[1] = clamp(mc.state[1] + mc.state[2], mc.min_position, mc.max_position)

    if mc.state[1] == mc.min_position && mc.state[2] < 0
        mc.state[2] = 0
    end

    features = [action^2, terminated(mc)]

    if scalar
        return features' * weights(mc)
    else
        return features
    end
end

env = VecEnv(; n_envs=8) do 
    MountainCar()
end |> LoggingWrapper

env = ObsNorm(; env)

solver = PPOSolver(; env, lr=1f-4, ac=ActorCritic(env; squash=true))

solve(solver)

function plotLoggingWrapper(env)
    info_dict = get_info(env)["LoggingWrapper"]
    p1 = plot(info_dict["steps"], info_dict["reward"], label=false, ylabel="Reward")
    p2 = plot(info_dict["steps"], info_dict["episode_length"], label=false, ylabel="Episode Length")
    plot(p1,p2; layout=(2,1))
end

plotLoggingWrapper(env)

solver.ac.log_std

