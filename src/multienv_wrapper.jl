module MultiEnvWrappers

using CommonRLInterface
using ..MultiEnv
using ..Spaces: NumericArraySpace

using Statistics
using Parameters: @with_kw

export
    AbstractMultiWrapper,
    wrapped_env,
    unwrapped,
    ObsNorm,
    RewTransform,
    LoggingWrapper,
    TransformWrapper,
    RewNorm

"""
    AbstractWrapper
"""
abstract type AbstractMultiWrapper <: AbstractMultiEnv end

"""
    wrapped_env(env)

Return the wrapped environment for an AbstractWrapper.

This is a *required function* that must be provided by every AbstractWrapper.

See also [`unwrapped`](@ref).
"""
function wrapped_env end

"""
    unwrapped(env)

Return the environment underneath all layers of wrappers.

See also [wrapped_env`](@ref).
"""
unwrapped(env::AbstractMultiWrapper) = unwrapped(wrapped_env(env))
unwrapped(env::AbstractMultiEnv) = env

macro forward_to_wrapped(f)
    return :($f(w::AbstractMultiWrapper, args...; kwargs...) = $f(wrapped_env(w), args...; kwargs...))
end

@forward_to_wrapped CommonRLInterface.reset!
# @forward_to_wrapped CommonRLInterface.actions
@forward_to_wrapped CommonRLInterface.observe
@forward_to_wrapped CommonRLInterface.act!
@forward_to_wrapped CommonRLInterface.terminated

@forward_to_wrapped CommonRLInterface.render
@forward_to_wrapped CommonRLInterface.state
@forward_to_wrapped CommonRLInterface.setstate!
@forward_to_wrapped CommonRLInterface.valid_actions
@forward_to_wrapped CommonRLInterface.valid_action_mask
# @forward_to_wrapped CommonRLInterface.observations
# not straightforward to provide clone

@forward_to_wrapped CommonRLInterface.players
@forward_to_wrapped CommonRLInterface.player
@forward_to_wrapped CommonRLInterface.all_act!
@forward_to_wrapped CommonRLInterface.all_observe
@forward_to_wrapped CommonRLInterface.UtilityStyle

@forward_to_wrapped Base.length
@forward_to_wrapped MultiEnv.single_observations
@forward_to_wrapped MultiEnv.single_actions

CommonRLInterface.provided(f::Function, w::AbstractMultiWrapper, args...) = provided(f, wrapped_env(w), args...)
CommonRLInterface.provided(::typeof(CommonRLInterface.clone), w::AbstractMultiWrapper, args...) = false

""" 
RunningStats
"""
mutable struct RunningStats
    k::Integer
    const M::AbstractArray
    const S::AbstractArray
end
RunningStats(T, sz) = RunningStats(0, zeros(T, sz), zeros(T, sz))
function (rs::RunningStats)(x)
    rs.k += 1
    if rs.k == 1
        rs.M .= x
    else
        x_diff = x .- rs.M
        rs.M .+= x_diff / rs.k
        rs.S .+= x_diff .* (x .- rs.M)
    end
    nothing
end
Statistics.mean(rs::RunningStats) = copy(rs.M)
Statistics.var(rs::RunningStats) = (rs.k==1) ? fill!(similar(rs.S), 0) : rs.S/(rs.k - 1)


""" 
ObsNorm
"""
struct ObsNorm <: AbstractMultiWrapper
    env::AbstractMultiEnv
    obs_stats::RunningStats
    s_lim::Number
end
wrapped_env(e::ObsNorm) = e.env

function ObsNorm(env; s_lim=10.0)
    space = single_observations(env)
    return ObsNorm(env, RunningStats(eltype(space), size(space)), abs(s_lim))
end

function CommonRLInterface.observe(wrap::ObsNorm)
    s_in = observe(wrap.env)

    for s in s_in
        wrap.obs_stats(s)
    end

    for s in s_in
        s .-= mean(wrap.obs_stats)
        s ./= sqrt.(var(wrap.obs_stats) .+ 1f-8)
        clamp!(s, -wrap.s_lim, wrap.s_lim)
    end

    return s_in
end


"""
RewNorm
"""
@with_kw struct RewNorm <: AbstractMultiWrapper
    env::AbstractMultiEnv
    rew_stats::RunningStats = RunningStats(Float32, 1)
    returns::Vector = zeros(Float32, length(env))
    gamma::Float32 = 1
    epsilon::Float32 = 1f-8
end
wrapped_env(e::RewNorm) = e.env

function CommonRLInterface.act!(wrap::RewNorm, a)
    r = act!(wrapped_env(wrap), a)
    wrap.returns .= r .+ wrap.gamma * .!terminated(wrapped_env(wrap)) .* wrap.returns
    wrap.rew_stats.(wrap.returns)
    return r ./ sqrt.(var(wrap.rew_stats) .+ wrap.epsilon)
end


"""
LoggingWrapper
"""
@with_kw struct LoggingWrapper <: AbstractMultiWrapper
    env::AbstractMultiEnv
    current_step::Vector{Int} = zeros(Int, length(env))
    total_step = [0]
    current_reward::Vector{Float32} = zeros(Int, length(env))
    step::Vector{Int} = Int[]
    episode_length::Vector{Int} = Int[]
    reward::Vector{Float32} = Float32[]
end
LoggingWrapper(env) = LoggingWrapper(; env)

wrapped_env(w::LoggingWrapper) = w.env

function CommonRLInterface.act!(w::LoggingWrapper, a)
    r = act!(w.env, a)

    w.current_reward .+= r

    w.current_step .+= 1
    w.total_step[] += length(w.env)

    dones = terminated(w.env)
    n_done = count(dones)
    if n_done > 0
        append!(w.step, fill(w.total_step[], n_done))
        append!(w.episode_length, w.current_step[dones])
        append!(w.reward, w.current_reward[dones])
        w.current_step[dones] .= 0
        w.current_reward[dones] .= 0
    end

    return r
end


"""
TransformWrapper
"""
@with_kw struct TransformWrapper <: AbstractMultiWrapper
    env::AbstractMultiEnv
    action_fun::Function = identity
    single_actions::NumericArraySpace = single_actions(env)
    observation_fun::Function = identity
    single_observations::NumericArraySpace = single_observations(env)
end
wrapped_env(e::TransformWrapper) = e.env

MultiEnv.single_actions(e::TransformWrapper) = e.single_actions
CommonRLInterface.act!(e::TransformWrapper, a) = act!(e.env, a) .|> e.action_fun

MultiEnv.single_observations(e::TransformWrapper) = e.single_observations
CommonRLInterface.observe(e::TransformWrapper) = observe(e.env) |> e.observation_fun

end