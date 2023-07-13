module MultiEnv

using CommonRLInterface
using ..Spaces

export
    AbstractMultiEnv,
    envs,
    single_actions,
    single_observations,
    VecEnv


const RL = CommonRLInterface

abstract type AbstractMultiEnv <: AbstractEnv end

function envs(::AbstractMultiEnv) end

Base.length(e::AbstractMultiEnv) = length(envs(e))

RL.reset!(e::AbstractMultiEnv) = reset!.(envs(e))
RL.observe(e::AbstractMultiEnv) = observe.(envs(e))
RL.terminated(e::AbstractMultiEnv) = terminated.(envs(e))

RL.actions(e::AbstractMultiEnv) = MultiAgentArraySpace(single_actions(e), length(e))
RL.observations(e::AbstractMultiEnv) = MultiAgentArraySpace(single_observations(e), length(e))

single_actions(e::AbstractMultiEnv) = actions(first(envs(e)))
single_observations(e::AbstractMultiEnv) = observations(first(envs(e)))


struct VecEnv <: AbstractMultiEnv
    envs::Vector{AbstractEnv}
    dones::Vector{Bool}
    auto_reset::Bool
end

function VecEnv(env_fcn::Function; n_envs=1, auto_reset=true)
    envs = [env_fcn() for _ in 1:n_envs]
    dones = terminated.(envs)
    return VecEnv(envs, dones, auto_reset)
end

envs(e::VecEnv) = e.envs

function RL.act!(e::VecEnv, a)
    r = vecEnvAct!(e, a)
    e.dones .= terminated.(envs(e))
    if e.auto_reset
        reset!.(envs(e)[e.dones])
    end
    return r
end
function vecEnvAct!(e::VecEnv, a::Vector)
    @assert length(envs(e))==length(a) "Number of actions does not match number of envs."
    r = act!.(envs(e), a)
    return r
end
function vecEnvAct!(e::VecEnv, a::Array)
    dims = ndims(a)
    @assert length(envs(e))==size(a, dims) "Number of actions does not match number of envs."
    r = act!.(envs(e), eachslice(a; dims))
    return r
end 

RL.terminated(e::VecEnv) = e.dones



end