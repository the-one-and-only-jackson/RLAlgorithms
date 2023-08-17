module MultiEnv

using CommonRLInterface
using Parameters: @with_kw
using ..Spaces

export
    AbstractMultiEnv,
    envs,
    single_actions,
    single_observations,
    VecEnv


const RL = CommonRLInterface

abstract type AbstractMultiEnv <: AbstractEnv end

RL.actions(e::AbstractMultiEnv) = MultiAgentArraySpace(single_actions(e), length(e))
RL.observations(e::AbstractMultiEnv) = MultiAgentArraySpace(single_observations(e), length(e))


@with_kw struct VecEnv <: AbstractMultiEnv
    envs::Vector{<:AbstractEnv}
    dones::Vector{Bool} = fill(false, length(envs))
    auto_reset::Bool = true
    batch::Bool = false
end

function VecEnv(env_fcn::Function; n_envs=1, kw_args...)
    envs = [env_fcn() for _ in 1:n_envs]
    return VecEnv(; envs, kw_args...)
end

Base.length(e::VecEnv) = length(e.envs)

function RL.reset!(e::VecEnv)
    reset!.(e.envs)
    e.dones .= terminated.(e.envs)
    nothing
end

function RL.act!(e::VecEnv, a)
    r = vecEnvAct!(e, a)
    e.dones .= terminated.(e.envs)
    if e.auto_reset
        reset!.(e.envs[e.dones])
    end
    return r
end
function vecEnvAct!(e::VecEnv, a::AbstractVector)
    @assert length(e.envs)==length(a) "Number of actions does not match number of envs."
    r = act!.(e.envs, a)
    return r
end
function vecEnvAct!(e::VecEnv, a::AbstractArray)
    dims = ndims(a)
    @assert length(e.envs)==size(a, dims) "Number of actions does not match number of envs."
    r = act!.(e.envs, eachslice(a; dims))
    return r
end 

RL.terminated(e::VecEnv) = e.dones
RL.observe(e::VecEnv) = observe.(e.envs)

single_actions(e::VecEnv) = actions(first(e.envs))
single_observations(e::VecEnv) = observations(first(e.envs))

RL.valid_action_mask(e::VecEnv) = valid_action_mask.(e.envs)
# Note: setup "forward to wrapped" and provided

end